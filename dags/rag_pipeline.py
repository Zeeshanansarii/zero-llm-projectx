from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from rag_app import index_data
from train import train_model
from fine_tuning import fine_tune_model

with DAG(
    'rag_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval='@weekly',
    catchup=False
) as dag:
    index_task = PythonOperator(
        task_id='index_data',
        python_callable=index_data,
        op_kwargs={'data_file': 'data.txt'}
    )
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={
            'data_file': 'data.txt',
            'vocab_size': 1000,
            'seq_len': 16,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'd_ff': 512,
            'epochs': 5,
            'batch_size': 32
        }
    )
    fine_tune_task = PythonOperator(
        task_id='fine_tune_advanced',
        python_callable=fine_tune_model,
        op_kwargs={
            'data_file': 'data.txt',
            'config': {
                'vocab_size': 1000,
                'seq_len': 16,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4,
                'd_ff': 512,
                'epochs': 5,
                'batch_size': 32,
                'accum_steps': 4,
                'learning_rate': 0.0001
            }
        }
    )
    
    index_task >> train_task >> fine_tune_task