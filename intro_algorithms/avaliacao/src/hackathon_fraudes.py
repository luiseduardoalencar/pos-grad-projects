"""
Hackathon Relâmpago - Caçadores de Fraudes
Autores: Luis Eduardo Alencar Melo & João Pedro Coelho Barbosa
Data: 06/04/2025
Objetivo: Detectar fraudes em lista de compras públicas utilizando Python.
"""

import pandas as pd
import re
from datetime import time

def carregar_dados(caminho_csv):
    df = pd.read_csv(r"C:\Users\luis\code\python\pos-grad\Introducao_a_algoritmos\avaliacao\data\public_servant_purchases.csv")
    df = df[df['nome_do_funcionario'].apply(lambda x: bool(re.match(r'^[A-Za-z\s]+$', str(x))))]
    df = df.dropna(subset=["nome_do_funcionario"])
    df['horario_da_compra'] = pd.to_datetime(df['data_da_compra']).dt.time
    df['dia_da_semana'] = pd.to_datetime(df['data_da_compra']).dt.day_name()
    return df

def detectar_compras_duplicadas(df):
    duplicates_date = df['data_da_compra'].duplicated()
    duplicates_true = df[duplicates_date].loc[:, ['data_da_compra', 'nome_do_funcionario', 'valor_em_real']]
    return duplicates_true.sort_values(by='data_da_compra')

def verificar_valores_suspeitos(df, limite=1000):
    return df[df["valor_em_real"] > limite]

def compras_fora_de_horario(df):
    return df[(df['horario_da_compra'] < time(8, 0)) | (df['horario_da_compra'] > time(18, 0))]

def organizar_por_servidor(df):
    return df.groupby("nome_do_funcionario")[['valor_em_real']].sum().sort_values(by='valor_em_real', ascending=False)

def gerar_relatorio(df):
    print("\n Compras Duplicadas:")
    print(detectar_compras_duplicadas(df))

    print("\n Compras com valor suspeito (> R$1000):")
    print(verificar_valores_suspeitos(df))

    print("\n Compras fora do horário comercial:")
    print(compras_fora_de_horario(df))

    print("\n Total de compras por servidor:")
    print(organizar_por_servidor(df))

if __name__ == "__main__":
    print("Iniciando detecção de fraudes...\n")
    caminho = "public_servant_purchases.csv"  
    compras = carregar_dados(caminho)
    gerar_relatorio(compras)
