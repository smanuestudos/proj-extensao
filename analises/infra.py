import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_PATH  = "microdados_censo_escolar_2024/dados/microdados_ed_basica_2024.csv"
OUT_DIR = "resultados"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH, encoding='latin1', sep=';', low_memory=False)

#Filtrar Curitiba 
df_ctba = df[df["NO_MUNICIPIO"].str.upper() == "CURITIBA"].copy()
print("Escolas em Curitiba:", len(df_ctba))

SAMPLE_N = 200
sample = df_ctba.sample(n=min(SAMPLE_N, len(df_ctba)), random_state=42)
sample.to_csv(os.path.join(OUT_DIR, "sample_curitiba.csv"), index=False, encoding="utf-8")
print("Amostra salva em:", os.path.join(OUT_DIR, "sample_curitiba.csv"))

#Colunas de interesse
cols_basica = ["IN_AGUA_POTAVEL", "IN_ESGOTO_REDE_PUBLICA", "IN_LIXO_SERVICO_COLETA", "IN_ENERGIA_REDE_PUBLICA"]
cols_ambientes = ["IN_BIBLIOTECA", "IN_BIBLIOTECA_SALA_LEITURA", "IN_LABORATORIO_INFORMATICA",
                  "IN_LABORATORIO_CIENCIAS", "IN_COZINHA", "IN_REFEITORIO", "IN_QUADRA_ESPORTES_COBERTA",
                  "IN_QUADRA_ESPORTES_DESCOBERTA", "IN_PARQUE_INFANTIL"]
cols_acess = ["IN_ACESSIBILIDADE_RAMPAS", "IN_ACESSIBILIDADE_PISOS_TATEIS", "IN_ACESSIBILIDADE_ELEVADOR",
              "IN_ACESSIBILIDADE_SINAL_VISUAL", "IN_ACESSIBILIDADE_SINAL_SONORO", "IN_ACESSIBILIDADE_SINAL_TATIL",
              "IN_ACESSIBILIDADE_CORRIMAO"]
cols_tec = ["IN_COMPUTADOR", "IN_EQUIP_MULTIMIDIA", "IN_EQUIP_LOUSA_DIGITAL",
            "IN_BANDA_LARGA", "IN_INTERNET", "IN_INTERNET_ALUNOS"]

def ensure_binary(df_local, col):
    if col not in df_local.columns:
        df_local[col] = 0
        return df_local[col]
    s = pd.to_numeric(df_local[col], errors='coerce').fillna(0)
    if s.dtype == object or s.isin([ "SIM","NÃO","NAO" ]).any():
        s = df_local[col].astype(str).str.upper().map({'SIM':1,'S':1,'1':1,'NÃO':0,'NAO':0,'N':0}).fillna(0).astype(int)
    return s.astype(int)

def compute_indicators(df_input, out_prefix):
    df_local = df_input.copy()
    # aplicar ensure_binary para cada coluna das listas
    for c in (cols_basica + cols_ambientes + cols_acess + cols_tec):
        df_local[c] = ensure_binary(df_local, c)

    df_local["INFRA_BASICA"] = df_local[cols_basica].mean(axis=1)
    df_local["AMBIENTES_EDUC"] = df_local[cols_ambientes].mean(axis=1)
    df_local["ACESSIBILIDADE"] = df_local[cols_acess].mean(axis=1)
    df_local["TECNOLOGIA"] = df_local[cols_tec].mean(axis=1)

    df_local["INDICE_GERAL_INFRA"] = df_local[["INFRA_BASICA","AMBIENTES_EDUC","ACESSIBILIDADE","TECNOLOGIA"]].mean(axis=1)

    cols_out = ["NO_ENTIDADE","CO_ENTIDADE","NO_MUNICIPIO","NO_BAIRRO"] + ["INFRA_BASICA","AMBIENTES_EDUC","ACESSIBILIDADE","TECNOLOGIA","INDICE_GERAL_INFRA"]
    out_csv = os.path.join(OUT_DIR, f"{out_prefix}_indicadores.csv")
    df_local[cols_out].to_csv(out_csv, index=False, encoding="utf-8")
    print("Indicadores salvos em:", out_csv)

    plt.figure(figsize=(8,5))
    df_local["INDICE_GERAL_INFRA"].hist(bins=20)
    plt.title(f"Distribuição Índice Geral Infra - {out_prefix}")
    plt.xlabel("Índice (0 a 1)")
    plt.ylabel("Quantidade de escolas")
    out_png = os.path.join(OUT_DIR, f"{out_prefix}_hist_indice.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("Gráfico salvo em:", out_png)

# rodar na amostra
compute_indicators(sample, "sample_curitiba")

# compute_indicators(df_ctba, "curitiba_completo")

print("Processo concluído. Verifique a pasta 'resultados'.")