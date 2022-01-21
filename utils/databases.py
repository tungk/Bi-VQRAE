import sqlite3
import psycopg2


def connect_database(host='fl-cs-chenjuan1.srv.aau.dk', database_name='RESULTS', database_type='POSTGRESQL'):
    if database_type=='POSTGRESQL':
        try:
            '''Need to create database manually'''
            conn = psycopg2.connect(host=host,
                                    port="5432",
                                    database=database_name,
                                    user="tung",
                                    password="123456")
            return conn
        except:
            conn = sqlite3.connect('./{}.db'.format(database_name))
            return conn
    if database_type=='SQLITE':
        conn = sqlite3.connect('./{}.db'.format(database_name))
        return conn


def create_schema(connection, table_name):
    create_sql = """CREATE TABLE IF NOT EXISTS {} (
    model_name CHAR(20) NOT NULL, pid CHAR(20) NOT NULL, dataset CHAR(20) NOT NULL, file_name CHAR(20) NOT NULL, settings CHAR(500), 
    TN_SD CHAR(10), FP_SD CHAR(10), FN_SD CHAR(10), TP_SD CHAR(10), precision_SD CHAR(10), recall_SD CHAR(10), fbeta_SD CHAR(10), cks_SD CHAR(10), 
    TN_MAD CHAR(10), FP_MAD CHAR(10), FN_MAD CHAR(10), TP_MAD CHAR(10), precision_MAD CHAR(10), recall_MAD CHAR(10), fbeta_MAD CHAR(10), cks_MAD CHAR(10), 
    TN_IQR CHAR(10), FP_IQR CHAR(10), FN_IQR CHAR(10), TP_IQR CHAR(10), precision_IQR CHAR(10), recall_IQR CHAR(10), fbeta_IQR CHAR(10), cks_IQR CHAR(10), 
    pr_auc CHAR(10), roc_auc CHAR(10), 
    best_TN_SD CHAR(10), best_FP_SD CHAR(10), best_FN_SD CHAR(10), best_TP_SD CHAR(10), best_precision_SD CHAR(10), best_recall_SD CHAR(10), best_fbeta_SD CHAR(10), cks_SD CHAR(10), 
    best_TN_MAD CHAR(10), best_FP_MAD CHAR(10), best_FN_MAD CHAR(10), best_TP_MAD CHAR(10), best_precision_MAD CHAR(10), best_recall_MAD CHAR(10), best_fbeta_MAD CHAR(10), cks_MAD CHAR(10), 
    best_TN_IQR CHAR(10), best_FP_IQR CHAR(10), best_FN_IQR CHAR(10), best_TP_IQR CHAR(10), best_precision_IQR CHAR(10), best_recall_IQR CHAR(10), best_fbeta_IQR CHAR(10), cks_IQR CHAR(10), 
    best_pr_auc CHAR(10), best_roc_auc CHAR(10), training_time CHAR(10), testing_time CHAR(10), memory CHAR(10), 
    PRIMARY KEY (model_name, pid, dataset, file_name));""".format(table_name)
    cursor_obj = connection.cursor()
    cursor_obj.execute(create_sql)
    cursor_obj.close()
    connection.commit()
    connection.close()

def insert_sample(connection, table_name, sample):
    inset_sql = """INSERT OR REPLACE INTO {} (
    model_name, pid, settings, dataset, file_name, 
    TN_SD, FP_SD, FN_SD, TP_SD, precision_SD, recall_SD, fbeta_SD, cks_SD,
    TN_MAD, FP_MAD, FN_MAD, TP_MAD, precision_MAD, recall_MAD, fbeta_MAD, cks_MAD,
    TN_IQR, FP_IQR, FN_IQR, TP_IQR, precision_IQR, recall_IQR, fbeta_IQR, cks_IQR,
    pr_auc, roc_auc, 
    best_TN_SD, best_FP_SD, best_FN_SD, best_TP_SD, best_precision_SD, best_recall_SD, best_fbeta_SD, best_cks_SD,
    best_TN_MAD, best_FP_MAD, best_FN_MAD, best_TP_MAD, best_precision_MAD, best_recall_MAD, best_fbeta_MAD, best_cks_MAD,
    best_TN_IQR, best_FP_IQR, best_FN_IQR, best_TP_IQR, best_precision_IQR, best_recall_IQR, best_fbeta_IQR, best_cks_IQR,
    best_pr_auc, best_roc_auc,
    training_time, testing_time, memory) VALUES('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', 
    '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', 
    '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', 
    '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');""".format(
        table_name, sample["model_name"], sample["pid"], sample["settings"], sample["dataset"], sample["file_name"],
        sample["TN_SD"], sample["FP_SD"], sample["FN_SD"], sample["TP_SD"], sample["precision_SD"], sample["recall_SD"],
        sample["fbeta_SD"], sample["cks_SD"], sample["TN_MAD"], sample["FP_MAD"], sample["FN_MAD"], sample["TP_MAD"],
        sample["precision_MAD"], sample["recall_MAD"], sample["fbeta_MAD"], sample["cks_MAD"], sample["TN_IQR"],
        sample["FP_IQR"], sample["FN_IQR"], sample["TP_IQR"], sample["precision_IQR"], sample["recall_IQR"],
        sample["fbeta_IQR"], sample["cks_IQR"], sample["pr_auc"], sample["roc_auc"], sample["best_TN_SD"],
        sample["best_FP_SD"], sample["best_FN_SD"], sample["best_TP_SD"], sample["best_precision_SD"],
        sample["best_recall_SD"], sample["best_fbeta_SD"], sample["best_cks_SD"], sample["best_TN_MAD"],
        sample["best_FP_MAD"], sample["best_FN_MAD"], sample["best_TP_MAD"], sample["best_precision_MAD"],
        sample["best_recall_MAD"], sample["best_fbeta_MAD"], sample["best_cks_MAD"], sample["best_TN_IQR"],
        sample["best_FP_IQR"], sample["best_FN_IQR"], sample["best_TP_IQR"], sample["best_precision_IQR"],
        sample["best_recall_IQR"], sample["best_fbeta_IQR"], sample["best_cks_IQR"], sample["best_pr_auc"],
        sample["best_roc_auc"], sample["training_time"], sample["testing_time"], sample["memory"])
    cursor_obj = connection.cursor()
    cursor_obj.execute(inset_sql)
    cursor_obj.close()
    connection.commit()


