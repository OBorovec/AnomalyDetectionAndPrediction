import json, yaml
import os
from enum import Enum

import requests, zipfile, io

import datetime
import dateutil.parser as dp

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

import ipywidgets as widgets
from IPython.display import display


DATASET_PROBLEM_DETECTION_PRIOR = 10
DATASET_PROBLEM_DETECTION_POST = 10

DATASET_MAINTENANCE_PRIOR = 10
DATASET_MAINTENANCE_POST = 0

DATASET_URL = "https://drive.google.com/open?id=1GhUU_pdRqB5ad4TjgaAhC0UFbvyRrHXb"


class DATASETS(Enum):
    DATASET_TEST = os.path.dirname(os.path.abspath(__file__)) + "/dataset_test/"
    DATASET_1 = os.path.dirname(os.path.abspath(__file__)) + "/dataset_1/"
    DATASET_2 = os.path.dirname(os.path.abspath(__file__)) + "/dataset_2/"


class APPILCATION(Enum):
    ELASTICSEARCH = "elasticsearch"
    LOGSTASH = "logstash"
    KIBANA = "kibana"
    FILEBEAT = "filebeat"
    NGINX = "nginx"
    RABBITMQ = "rabbitmq"
    CURATOR = "curator"


class DATASET_EVENTS(Enum):
    LOG = "log"
    MONITORING_ES_CLUSTER = "monitoring_es_cluster"
    MONITORING_LOGSTASH = "monitoring_logstash"
    MONITORING_KIBANA = "monitoring_kibana"


class DatasetProducer:

    def __init__(self, dataset_name, window=1,
                 include_artificial_anomalies=False,
                 separate_hosts=False,
                 single_application=None):
        self.dataset = dataset_name
        self.window = window
        self.include_artificial_anomalies = include_artificial_anomalies
        self.separate_hosts = separate_hosts
        self.single_application = single_application

        self.dataset_properties = self.__load_dataset_properties()
        self.dataset_files = self.__load_dataset_files()
        self.dataset_files_training, self.dataset_files_run = self._split_dataset_files()

        self.evaluation = DatasetEvaluation(self)

    def get_evaluation(self):
        return DatasetEvaluation(self)

    def emulate_log_count_matrix(self, iwiget=False, training=False):
        file_scope = self.dataset_files_training if training else self.dataset_files_run
        max_features = self.dataset_properties["cardinality_log_key"] * self.dataset_properties["cardinality_hostname"] \
            if self.separate_hosts \
            else self.dataset_properties["cardinality_log_key"]
        window_log_pattern_key_count = np.zeros(max_features, dtype=np.int)
        free_index = 0
        feature_index_dict = {}
        sequence_index = []
        for log in self.__iterate_log(files=file_scope, iwiget=iwiget):
            log_timestamp = get_log_timestamp(log)
            log_pattern_key = get_log_pattern_key(log)
            if self.separate_hosts:
                log_pattern_key += log["hostname"]
            if log_pattern_key not in feature_index_dict:
                feature_index_dict[log_pattern_key] = free_index
                free_index += 1
            feature_index = feature_index_dict[log_pattern_key]
            sequence_index.append(feature_index)
            window_log_pattern_key_count[feature_index] += 1
            if len(sequence_index) > self.window:
                window_log_pattern_key_count[sequence_index.pop(0)] -= 1
            if len(sequence_index) == self.window:
                yield log_timestamp, csr_matrix(window_log_pattern_key_count)

    def emulate_log_sequence(self, iwiget=False, training=False):
        sequence = []
        file_scope = self.dataset_files_training if training else self.dataset_files_run
        for log in self.__iterate_log(iwiget=iwiget, files=file_scope):
            sequence.append(log)
            if len(sequence) > self.window:
                sequence.pop(0)
            if len(sequence) == self.window:
                yield sequence

    def emulate_log_sequence_matrix(self):
        pass

    def __load_dataset_files(self):
        return [self.dataset.value + file
                for file in sorted(os.listdir(str(self.dataset.value)))
                if file.endswith(".json")]

    def _split_dataset_files(self):
        safe_bound_str = dp.parse(self.dataset_properties["safe_bound"]).strftime("%Y-%m-%d-%H-%M") + ".json"
        safe_bound_file_idx = None
        for idx, file in enumerate(self.dataset_files):
            if file.endswith(safe_bound_str):
                safe_bound_file_idx = idx
                break
        return self.dataset_files[:safe_bound_file_idx+1], self.dataset_files[safe_bound_file_idx+1:]

    def _load_dataset_labels(self):
        df = pd.read_csv(self.dataset.value + "labels.csv", header=None, sep=";")
        df.columns = ["from", "to", "description"]
        df["to"].fillna(df["from"], inplace=True)
        df["from"] = df['from'].apply(dp.parse) - datetime.timedelta(minutes=DATASET_PROBLEM_DETECTION_PRIOR)
        df["to"] = df['to'].apply(dp.parse) + datetime.timedelta(minutes=DATASET_PROBLEM_DETECTION_POST)
        df["detected"] = False
        return df

    def _load_dataset_maintenance(self):
        df = pd.read_csv(self.dataset.value + "maintenance.csv", header=None, sep=";")
        df.columns = ["time", "description"]
        df["time"] = df['time'].apply(dp.parse)
        df["detected"] = False
        return df

    def __load_dataset_properties(self):
        with open(self.dataset.value + "properties.yaml") as property_file:
            return yaml.load(property_file)

    def __load_dataset_segment(self, scope_file):
        segment = json.load(scope_file)
        logs = segment.get('log') or []
        monitoring_logstash = segment.get('monitoring-logstash') or []
        monitoring_es_cluster = segment.get('monitoring-es-cluster') or []
        monitoring_kibana = segment.get('monitoring-kibana') or []
        return logs, monitoring_logstash, monitoring_es_cluster, monitoring_kibana

    def __iterate_log(self, files, iwiget=False):
        if iwiget:
            proccess_main = widgets.IntProgress(
                value=0,
                min=0,
                max=len(files),
                step=1,
                description='Files:',
                bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
                orientation='horizontal'
            )
            display(proccess_main)
            proccess_segment = widgets.IntProgress(
                value=0,
                min=0,
                max=10,
                step=1,
                description='Segments:',
                bar_style='',
                orientation='horizontal'
            )
            display(proccess_segment)
        for file_path in files:
            if iwiget:
                proccess_main.value += 1
            with open(file_path) as scope_file:
                logs, _, _, _ \
                    = self.__load_dataset_segment(scope_file)
                if iwiget:
                    proccess_segment.value, proccess_segment.max = 0, len(logs)
                for log in logs:
                    if iwiget:
                        proccess_segment.value += 1
                    yield log
        if iwiget:
            proccess_main.close()
            proccess_segment.close()
        print("Done.")

    def __iterate_all(self):
        for file_path in self.dataset_files:
            with open(file_path) as scope_file:
                logs, monitoring_logstash, monitoring_es_cluster, monitoring_kibana \
                    = self.__load_dataset_segment(scope_file)
                #TODO: emulate on every incoming event

    def __slice_monitoring(self, monitoring_queue, target_ts):
        monitoring_pop = []
        while len(monitoring_queue) > 0 \
                and dp.parse(monitoring_queue[0]["@timestamp"]) < target_ts:
            monitoring_pop.append(monitoring_queue.pop(0))
        return monitoring_pop

    def submit(self, ts):
        return self.evaluation.submit(ts)

    def evaluate(self):
        self.evaluation.evaluate()


class DatasetEvaluation:

    def __init__(self, ds):
        self.submits = 0
        self.matched = 0
        self.maintenance = 0
        self.fp = 0
        self.score = 0
        self.dataset_labels = ds._load_dataset_labels()
        self.dataset_maintenance = ds._load_dataset_maintenance()

    def submit(self, ts):
        if isinstance(ts, str):
            ts = dp.parse(ts)
        self.submits += 1
        if len(self.dataset_labels[
               (self.dataset_labels["from"] < ts) & (ts < self.dataset_labels["to"])
               ]) > 0:
            # True problem detected
            self.dataset_labels.loc[
                (self.dataset_labels["from"] < ts) & (ts < self.dataset_labels["to"]), "detected"
            ] = True
            self.matched += 1
            return 1
        else:
            left_bound = ts - datetime.timedelta(minutes=DATASET_MAINTENANCE_PRIOR)
            right_bound = ts + datetime.timedelta(minutes=DATASET_MAINTENANCE_POST)
            if sum(self.dataset_maintenance["time"].between(left_bound, right_bound)) > 0:
                self.maintenance += 1
                self.dataset_maintenance.loc[
                    self.dataset_maintenance["time"].between(left_bound, right_bound), "detected"
                ] = True
                return 0
            else:
                self.fp += 1
                return -1

    def evaluate(self):
        print("_____________________________________________________")
        print("Problems detected: {} of {}".format(self.dataset_labels["detected"].sum(), len(self.dataset_labels)))
        print("FP count: {}".format(self.fp))
        print("Maintenance hits: {}".format(self.maintenance))
        print("Total score: {}".format(self.score))
        print("_____________________________________________________")


def get_log_pattern_key(log):
    pattern_key = str(log.get("application")) + \
                  str(log.get("type")) + \
                  str(log.get("logger")) + \
                  str(log.get("level")) +\
                  str(log.get("message_pattern"))
    return pattern_key


def get_log_timestamp(log):
    return dp.parse(log["@timestamp"])


def download():
    r = requests.get(DATASET_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


if __name__ == '__main__':
    ds = DatasetProducer(DATASETS.DATASET_TEST, window=1)
    logs_seen = 0
    for log_list in ds.emulate_log_sequence(iwiget=False, training=True):
        logs_seen += 1
    for log_list in ds.emulate_log_sequence(iwiget=False, training=False):
        logs_seen += 1
    print("Have seen all logs: {}".format(logs_seen == ds.dataset_properties["logs_total"]))
    print("Recognised FP example: {}".format(-1 == ds.submit(dp.parse("2019-01-10T0:22:00.0Z"))))
    print("Hit to maintenance 30s prior: {}".format(0 == ds.submit(dp.parse("2019-01-10T0:25:00.0Z") - datetime.timedelta(seconds=30))))
    print("Hit to maintenance: {}".format(0 == ds.submit(dp.parse("2019-01-10T0:25:00.0Z"))))
    print("Hit to maintenance 4 min post: {}".format(0 == ds.submit(dp.parse("2019-01-10T0:25:00.0Z") + datetime.timedelta(minutes=4))))
    print("Recognise TP 4 min prior: {}".format(1 == ds.submit(dp.parse("2019-01-10T0:39:00.0Z") - datetime.timedelta(minutes=4))))
    print("Recognised TP: {}".format(1 == ds.submit(dp.parse("2019-01-10T0:39:30.0Z"))))
    print("Recognise TP 5 min post: {}".format(1 == ds.submit(dp.parse("2019-01-10T0:40:30.0Z") + datetime.timedelta(minutes=5))))
    print("Recognise TP 9 min post: {}".format(1 == ds.submit(dp.parse("2019-01-10T0:40:30.0Z") + datetime.timedelta(minutes=9))))
    print("Correct FP after 11 mins: {}".format(-1 == ds.submit(dp.parse("2019-01-10T0:40:30.0Z") + datetime.timedelta(minutes=11))))
    ds.evaluate()
