# Benchmark robusto de PRNGs para IC
# Compara:
# 1) GNPA (adaptado do TCC)
# 2) Mersenne Twister (random padrão do Python)
# 3) Gerador caótico recorrente simples (tanh, 2 neurônios)
#
# Mede:
# - tempo de geração
# - tempo por número gerado
# - qui-quadrado (64 classes)
# - p-valor
# - classificação
#
# Também coleta:
# - dados do sistema
# - memória do processo
# - load average
# - timestamp
#
# Requisitos:
# pip3 install scipy psutil
#
# Como rodar:
# python3 benchmark_prng_repeticoes.py

import csv
import math
import os
import platform
import random
import socket
import time
from datetime import datetime
from statistics import mean, median

try:
    import psutil
except ImportError:
    print("Instale psutil antes: pip3 install psutil")
    raise

try:
    from scipy.stats import chi2
except ImportError:
    print("Instale scipy antes: pip3 install scipy")
    raise


# =========================
# CONFIGURAÇÕES DO TESTE
# =========================
SAMPLE_SIZES = [1000, 10000, 100000, 1000000]
REPEATS = 10
SEED = 123456789
BINS = 64
MAX_UINT32 = 2**32
ALPHA = 0.05

# True = mantém comportamento dependente do relógio
# False = usa modo determinístico interno
GNPA_USE_REAL_TIME = False

DETAILS_CSV = "resultados_detalhados_prng.csv"
SUMMARY_CSV = "resumo_prng.csv"
ENV_CSV = "ambiente_execucao.csv"


# =========================
# GERADOR GNPA
# =========================
class GNPA:
    def __init__(self, seed, use_real_time=False):
        self.seed = seed & 0xFFFFFFFF
        self.use_real_time = use_real_time
        self.internal_counter = (seed ^ 0xA5A5A5A5) & 0xFFFFFFFF

        self.sbox = {
            0x0: 0xE, 0x1: 0x4, 0x2: 0xD, 0x3: 0x1,
            0x4: 0x2, 0x5: 0xF, 0x6: 0xB, 0x7: 0x8,
            0x8: 0x3, 0x9: 0xA, 0xA: 0x6, 0xB: 0xC,
            0xC: 0x5, 0xD: 0x9, 0xE: 0x0, 0xF: 0x7,
        }

        self.pbox_map = [
            13, 10, 4, 11, 6, 3, 23, 1,
            9, 7, 15, 25, 20, 18, 28, 0,
            22, 29, 8, 17, 5, 27, 2, 16,
            31, 26, 12, 19, 24, 30, 14, 21,
        ]

    def number_to_bits(self, num):
        return format(num & 0xFFFFFFFF, "032b")

    def bits_to_number(self, bits):
        return int(bits, 2) & 0xFFFFFFFF

    def xor(self, a, b):
        return (a ^ b) & 0xFFFFFFFF

    def permute(self, value):
        bits = self.number_to_bits(value)
        output = "".join(bits[i] for i in self.pbox_map)
        return self.bits_to_number(output)

    def substitute(self, value):
        bits = self.number_to_bits(value)
        output = ""
        for i in range(0, 32, 4):
            nibble = int(bits[i:i + 4], 2)
            output += format(self.sbox[nibble], "04b")
        return self.bits_to_number(output)

    def produzir_congruencia(self):
        if self.use_real_time:
            now = datetime.now()
            alpha = int(f"{now.year}{now.month:02d}{now.day:02d}") & 0xFFFFFFFF
            beta = int(f"{now.hour:02d}{now.minute:02d}{now.second:02d}") & 0xFFFFFFFF
            gamma = int(f"{now.microsecond:06d}{time.perf_counter_ns() % 1000:03d}") & 0xFFFFFFFF
        else:
            self.internal_counter = (1664525 * self.internal_counter + 1013904223) & 0xFFFFFFFF
            alpha = (0x9E3779B9 ^ self.internal_counter) & 0xFFFFFFFF
            beta = (0x85EBCA6B ^ ((self.internal_counter << 7) & 0xFFFFFFFF)) & 0xFFFFFFFF
            gamma = (0xC2B2AE35 ^ ((self.internal_counter >> 3) & 0xFFFFFFFF)) & 0xFFFFFFFF

        x1 = ((gamma * alpha) + beta) % MAX_UINT32
        x2 = ((x1 * alpha) + beta) % MAX_UINT32
        x3 = ((x2 * alpha) + beta) % MAX_UINT32
        x4 = ((x3 * alpha) + beta) % MAX_UINT32
        return x1, x2, x3, x4

    def erfc(self, x):
        z = abs(x)
        t = 1 / (1 + 0.5 * z)
        poly = (
            -z * z - 1.26551223
            + 1.00002368 * t
            + 0.37409196 * t**2
            + 0.09678418 * t**3
            - 0.18628806 * t**4
            + 0.27886807 * t**5
            - 1.13520398 * t**6
            + 1.48851587 * t**7
            - 0.82215223 * t**8
            + 0.17087277 * t**9
        )
        r = t * math.exp(poly)
        return r if x >= 0 else 2 - r

    def monobit_test(self, value):
        bits = self.number_to_bits(value)
        s = sum(1 if b == "1" else -1 for b in bits)
        sobs = abs(s) / math.sqrt(32)
        return self.erfc(sobs / math.sqrt(2)) >= 0.01

    def gerar(self):
        current_seed = self.seed
        while True:
            x1, x2, x3, x4 = self.produzir_congruencia()
            estado = self.xor(current_seed, x1)
            estado = self.permute(estado)
            estado = self.xor(estado, x2)
            estado = self.substitute(estado)
            estado = self.xor(estado, x3)
            estado = self.xor(estado, x4)
            self.seed = estado
            current_seed = estado
            if self.monobit_test(estado):
                return estado


# =========================
# RNN CAÓTICA
# =========================
class ChaoticRNN:
    def __init__(self, seed):
        self.local_rng = random.Random(seed)
        self.x = 0.1234
        self.y = 0.5678
        self.w = [[1.92, -1.31], [1.17, 1.88]]

    def gerar(self):
        x_new = math.tanh(self.w[0][0] * self.x + self.w[0][1] * self.y)
        y_new = math.tanh(self.w[1][0] * self.x + self.w[1][1] * self.y)
        self.x, self.y = x_new, y_new
        value = int(((self.x + 1) / 2) * (MAX_UINT32 - 1))
        return value & 0xFFFFFFFF


# =========================
# UTILIDADES
# =========================
def bytes_to_mb(value):
    return value / (1024 * 1024)


def bytes_to_gb(value):
    return value / (1024 * 1024 * 1024)


def format_sample_size(size):
    return f"{size:,}".replace(",", ".")


def classify_pvalue(p_value, alpha=ALPHA):
    if p_value >= alpha:
        return "compatível com aleatoriedade uniforme"
    return "possível viés estatístico"


# =========================
# ESTATÍSTICAS
# =========================
def chi_square_test(numbers, bins=BINS):
    counts = [0] * bins
    width = MAX_UINT32 / bins

    for n in numbers:
        idx = min(int(n / width), bins - 1)
        counts[idx] += 1

    expected = len(numbers) / bins
    chi_sq = sum((c - expected) ** 2 / expected for c in counts)
    p_value = chi2.sf(chi_sq, bins - 1)
    return chi_sq, p_value


def summarize_group(rows):
    tempos = [r["tempo_s"] for r in rows]
    ns_por_numero = [r["ns_por_numero"] for r in rows]
    chis = [r["qui_quadrado"] for r in rows]
    pvals = [r["p_valor"] for r in rows]
    mem_rss_mb = [r["rss_processo_mb"] for r in rows]

    aprovados = sum(1 for r in rows if r["p_valor"] >= ALPHA)
    reprovados = len(rows) - aprovados

    return {
        "tempo_medio_s": mean(tempos),
        "tempo_mediano_s": median(tempos),
        "tempo_min_s": min(tempos),
        "tempo_max_s": max(tempos),
        "ns_por_numero_medio": mean(ns_por_numero),
        "ns_por_numero_min": min(ns_por_numero),
        "ns_por_numero_max": max(ns_por_numero),
        "qui2_medio": mean(chis),
        "qui2_min": min(chis),
        "qui2_max": max(chis),
        "p_medio": mean(pvals),
        "p_min": min(pvals),
        "p_max": max(pvals),
        "rss_medio_mb": mean(mem_rss_mb),
        "rss_min_mb": min(mem_rss_mb),
        "rss_max_mb": max(mem_rss_mb),
        "aprovados": aprovados,
        "reprovados": reprovados,
        "taxa_aprovacao_percent": 100.0 * aprovados / len(rows),
    }


# =========================
# COLETA DO AMBIENTE
# =========================
def get_system_snapshot():
    process = psutil.Process(os.getpid())

    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        load1, load5, load15 = (None, None, None)

    vm = psutil.virtual_memory()

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "sistema": platform.system(),
        "versao_sistema": platform.version(),
        "release": platform.release(),
        "arquitetura": platform.machine(),
        "processador": platform.processor(),
        "python_versao": platform.python_version(),
        "cpu_fisicas": psutil.cpu_count(logical=False),
        "cpu_logicas": psutil.cpu_count(logical=True),
        "memoria_total_gb": bytes_to_gb(vm.total),
        "memoria_disponivel_gb": bytes_to_gb(vm.available),
        "memoria_utilizada_percent": vm.percent,
        "rss_processo_mb": bytes_to_mb(process.memory_info().rss),
        "vms_processo_mb": bytes_to_mb(process.memory_info().vms),
        "loadavg_1min": load1,
        "loadavg_5min": load5,
        "loadavg_15min": load15,
    }


# =========================
# BENCHMARK
# =========================
def benchmark_generator(name, generator_factory, sample_sizes, repeats):
    results = []

    print(f"\n{'=' * 90}")
    print(f"GERADOR: {name}")
    print(f"{'=' * 90}")

    for size in sample_sizes:
        print(f"\nAmostra: {format_sample_size(size)}")
        print("-" * 90)

        group_rows = []

        for rep in range(1, repeats + 1):
            env_before = get_system_snapshot()
            gen = generator_factory()

            start = time.perf_counter()

            if hasattr(gen, "gerar"):
                numbers = [gen.gerar() for _ in range(size)]
            else:
                numbers = [gen.getrandbits(32) for _ in range(size)]

            elapsed = time.perf_counter() - start
            chi_sq, p_value = chi_square_test(numbers)
            quality = classify_pvalue(p_value)
            env_after = get_system_snapshot()

            ns_por_numero = (elapsed / size) * 1_000_000_000

            row = {
                "timestamp": env_after["timestamp"],
                "gerador": name,
                "repeticao": rep,
                "amostra": size,
                "tempo_s": elapsed,
                "ns_por_numero": ns_por_numero,
                "qui_quadrado": chi_sq,
                "p_valor": p_value,
                "qualidade": quality,
                "rss_processo_mb": env_after["rss_processo_mb"],
                "vms_processo_mb": env_after["vms_processo_mb"],
                "loadavg_1min": env_after["loadavg_1min"],
                "loadavg_5min": env_after["loadavg_5min"],
                "loadavg_15min": env_after["loadavg_15min"],
                "memoria_total_gb": env_after["memoria_total_gb"],
                "memoria_disponivel_gb": env_after["memoria_disponivel_gb"],
                "memoria_utilizada_percent": env_after["memoria_utilizada_percent"],
                "cpu_logicas": env_after["cpu_logicas"],
                "cpu_fisicas": env_after["cpu_fisicas"],
                "hostname": env_after["hostname"],
                "sistema": env_after["sistema"],
                "release": env_after["release"],
                "arquitetura": env_after["arquitetura"],
                "python_versao": env_after["python_versao"],
                "rss_processo_mb_antes": env_before["rss_processo_mb"],
            }

            group_rows.append(row)
            results.append(row)

            print(
                f"[Execução {rep:02d}] "
                f"Tempo: {elapsed:.6f} s | "
                f"ns/número: {ns_por_numero:.2f} | "
                f"Qui²: {chi_sq:.4f} | "
                f"p-valor: {p_value:.6f} | "
                f"RSS: {env_after['rss_processo_mb']:.2f} MB | "
                f"Load(1m): {env_after['loadavg_1min'] if env_after['loadavg_1min'] is not None else 'N/A'} | "
                f"Qualidade: {quality}"
            )

        summary = summarize_group(group_rows)

        print("\nResumo do bloco:")
        print(f"Tempo médio: {summary['tempo_medio_s']:.6f} s")
        print(f"Tempo mediano: {summary['tempo_mediano_s']:.6f} s")
        print(f"Tempo mínimo: {summary['tempo_min_s']:.6f} s")
        print(f"Tempo máximo: {summary['tempo_max_s']:.6f} s")
        print(f"ns por número (médio): {summary['ns_por_numero_medio']:.2f}")
        print(f"Faixa ns por número: {summary['ns_por_numero_min']:.2f} - {summary['ns_por_numero_max']:.2f}")
        print(f"Qui² médio: {summary['qui2_medio']:.4f}")
        print(f"Faixa Qui²: {summary['qui2_min']:.4f} - {summary['qui2_max']:.4f}")
        print(f"p médio: {summary['p_medio']:.6f}")
        print(f"Faixa p: {summary['p_min']:.6f} - {summary['p_max']:.6f}")
        print(f"RSS médio do processo: {summary['rss_medio_mb']:.2f} MB")
        print(f"Faixa RSS: {summary['rss_min_mb']:.2f} - {summary['rss_max_mb']:.2f} MB")
        print(f"Aprovações (p >= {ALPHA}): {summary['aprovados']}/{repeats}")
        print(f"Reprovações (p < {ALPHA}): {summary['reprovados']}/{repeats}")
        print(f"Taxa de aprovação: {summary['taxa_aprovacao_percent']:.1f}%")

    return results


# =========================
# CSV
# =========================
def save_detailed_csv(filepath, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "gerador",
            "repeticao",
            "amostra",
            "tempo_s",
            "ns_por_numero",
            "qui_quadrado",
            "p_valor",
            "qualidade",
            "rss_processo_mb",
            "vms_processo_mb",
            "rss_processo_mb_antes",
            "loadavg_1min",
            "loadavg_5min",
            "loadavg_15min",
            "memoria_total_gb",
            "memoria_disponivel_gb",
            "memoria_utilizada_percent",
            "cpu_fisicas",
            "cpu_logicas",
            "hostname",
            "sistema",
            "release",
            "arquitetura",
            "python_versao",
        ])

        for r in rows:
            writer.writerow([
                r["timestamp"],
                r["gerador"],
                r["repeticao"],
                r["amostra"],
                f"{r['tempo_s']:.9f}",
                f"{r['ns_por_numero']:.9f}",
                f"{r['qui_quadrado']:.9f}",
                f"{r['p_valor']:.9f}",
                r["qualidade"],
                f"{r['rss_processo_mb']:.6f}",
                f"{r['vms_processo_mb']:.6f}",
                f"{r['rss_processo_mb_antes']:.6f}",
                r["loadavg_1min"],
                r["loadavg_5min"],
                r["loadavg_15min"],
                f"{r['memoria_total_gb']:.6f}",
                f"{r['memoria_disponivel_gb']:.6f}",
                f"{r['memoria_utilizada_percent']:.2f}",
                r["cpu_fisicas"],
                r["cpu_logicas"],
                r["hostname"],
                r["sistema"],
                r["release"],
                r["arquitetura"],
                r["python_versao"],
            ])


def save_summary_csv(filepath, rows):
    grouped = {}
    for r in rows:
        key = (r["gerador"], r["amostra"])
        grouped.setdefault(key, []).append(r)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gerador",
            "amostra",
            "tempo_medio_s",
            "tempo_mediano_s",
            "tempo_min_s",
            "tempo_max_s",
            "ns_por_numero_medio",
            "ns_por_numero_min",
            "ns_por_numero_max",
            "qui2_medio",
            "qui2_min",
            "qui2_max",
            "p_medio",
            "p_min",
            "p_max",
            "rss_medio_mb",
            "rss_min_mb",
            "rss_max_mb",
            "aprovados",
            "reprovados",
            "taxa_aprovacao_percent",
        ])

        for (gerador, amostra), group_rows in grouped.items():
            s = summarize_group(group_rows)
            writer.writerow([
                gerador,
                amostra,
                f"{s['tempo_medio_s']:.9f}",
                f"{s['tempo_mediano_s']:.9f}",
                f"{s['tempo_min_s']:.9f}",
                f"{s['tempo_max_s']:.9f}",
                f"{s['ns_por_numero_medio']:.9f}",
                f"{s['ns_por_numero_min']:.9f}",
                f"{s['ns_por_numero_max']:.9f}",
                f"{s['qui2_medio']:.9f}",
                f"{s['qui2_min']:.9f}",
                f"{s['qui2_max']:.9f}",
                f"{s['p_medio']:.9f}",
                f"{s['p_min']:.9f}",
                f"{s['p_max']:.9f}",
                f"{s['rss_medio_mb']:.6f}",
                f"{s['rss_min_mb']:.6f}",
                f"{s['rss_max_mb']:.6f}",
                s["aprovados"],
                s["reprovados"],
                f"{s['taxa_aprovacao_percent']:.2f}",
            ])


def save_environment_csv(filepath):
    env = get_system_snapshot()

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["campo", "valor"])
        for key, value in env.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.6f}"])
            else:
                writer.writerow([key, value])


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Benchmark iniciado...\n")
    print(f"Amostras: {', '.join(format_sample_size(s) for s in SAMPLE_SIZES)}")
    print(f"Repetições por bloco: {REPEATS}")
    print(f"Número de classes (bins): {BINS}")
    print(f"Nível de significância (alpha): {ALPHA}")
    print(f"GNPA em modo temporal real: {GNPA_USE_REAL_TIME}")

    env = get_system_snapshot()
    print("\nAmbiente detectado:")
    print(f"Hostname: {env['hostname']}")
    print(f"Sistema: {env['sistema']} {env['release']}")
    print(f"Arquitetura: {env['arquitetura']}")
    print(f"Python: {env['python_versao']}")
    print(f"CPU físicas: {env['cpu_fisicas']}")
    print(f"CPU lógicas: {env['cpu_logicas']}")
    print(f"Memória total: {env['memoria_total_gb']:.2f} GB")
    print(f"Memória disponível: {env['memoria_disponivel_gb']:.2f} GB")
    print(f"Memória utilizada: {env['memoria_utilizada_percent']:.2f}%")
    print(f"RSS do processo: {env['rss_processo_mb']:.2f} MB")
    print(
        "Load average: "
        f"{env['loadavg_1min'] if env['loadavg_1min'] is not None else 'N/A'}, "
        f"{env['loadavg_5min'] if env['loadavg_5min'] is not None else 'N/A'}, "
        f"{env['loadavg_15min'] if env['loadavg_15min'] is not None else 'N/A'}"
    )

    all_results = []

    all_results.extend(
        benchmark_generator(
            "GNPA",
            lambda: GNPA(SEED, use_real_time=GNPA_USE_REAL_TIME),
            SAMPLE_SIZES,
            REPEATS,
        )
    )

    all_results.extend(
        benchmark_generator(
            "Mersenne Twister",
            lambda: random.Random(SEED),
            SAMPLE_SIZES,
            REPEATS,
        )
    )

    all_results.extend(
        benchmark_generator(
            "RNN caótica simples",
            lambda: ChaoticRNN(SEED),
            SAMPLE_SIZES,
            REPEATS,
        )
    )

    save_detailed_csv(DETAILS_CSV, all_results)
    save_summary_csv(SUMMARY_CSV, all_results)
    save_environment_csv(ENV_CSV)

    print("\nArquivos gerados:")
    print(f"- {DETAILS_CSV}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {ENV_CSV}")
    print("\nConcluído.")