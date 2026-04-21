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


# =========================================================
# CONFIGURAÇÕES
# =========================================================
SAMPLE_SIZES = [1000, 10000, 100000, 1000000]
REPEATS = 10
MASTER_SEED = 123456789
BINS = 64
MAX_UINT32 = 2**32
ALPHA = 0.05

USE_REAL_TIME = False

DETAILS_CSV = "detalhes_gnpa_mt.csv"
SUMMARY_CSV = "resumo_gnpa_mt.csv"
ENV_CSV = "ambiente_gnpa_mt.csv"
REPORT_TXT = "relatorio_automatico_gnpa_mt.txt"


# =========================================================
# DADOS FIXOS DO GNPA
# =========================================================
PBOX_MAP = [
    13, 10, 4, 11, 6, 3, 23, 1,
    9, 7, 15, 25, 20, 18, 28, 0,
    22, 29, 8, 17, 5, 27, 2, 16,
    31, 26, 12, 19, 24, 30, 14, 21,
]

SBOX = {
    0x0: 0xE, 0x1: 0x4, 0x2: 0xD, 0x3: 0x1,
    0x4: 0x2, 0x5: 0xF, 0x6: 0xB, 0x7: 0x8,
    0x8: 0x3, 0x9: 0xA, 0xA: 0x6, 0xB: 0xC,
    0xC: 0x5, 0xD: 0x9, 0xE: 0x0, 0xF: 0x7,
}


# =========================================================
# UTILIDADES
# =========================================================
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


def derive_seeds(master_seed, repeats):
    rng = random.Random(master_seed)
    return [rng.getrandbits(32) for _ in range(repeats)]


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


# =========================================================
# MONOBIT
# =========================================================
def erfc_approx(x):
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


def monobit_test_32(value):
    value = value & 0xFFFFFFFF
    bits_ones = bin(value).count("1")
    s = 2 * bits_ones - 32
    sobs = abs(s) / math.sqrt(32)
    return erfc_approx(sobs / math.sqrt(2)) >= 0.01


# =========================================================
# CBOX / CONGRUÊNCIA
# =========================================================
class GNPACongruenceMixin:
    def __init__(self, seed, use_real_time=False):
        self.seed = seed & 0xFFFFFFFF
        self.use_real_time = use_real_time
        self.internal_counter = (seed ^ 0xA5A5A5A5) & 0xFFFFFFFF

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


# =========================================================
# GNPA ORIGINAL ADAPTADO
# =========================================================
class GNPAOriginal(GNPACongruenceMixin):
    def number_to_bits(self, num):
        return format(num & 0xFFFFFFFF, "032b")

    def bits_to_number(self, bits):
        return int(bits, 2) & 0xFFFFFFFF

    def xor(self, a, b):
        return (a ^ b) & 0xFFFFFFFF

    def permute(self, value):
        bits = self.number_to_bits(value)
        output = "".join(bits[i] for i in PBOX_MAP)
        return self.bits_to_number(output)

    def substitute(self, value):
        bits = self.number_to_bits(value)
        output = ""
        for i in range(0, 32, 4):
            nibble = int(bits[i:i + 4], 2)
            output += format(SBOX[nibble], "04b")
        return self.bits_to_number(output)

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
            if monobit_test_32(estado):
                return estado


# =========================================================
# GNPA OTIMIZADO BITWISE
# =========================================================
class GNPAOptimized(GNPACongruenceMixin):
    def __init__(self, seed, use_real_time=False):
        super().__init__(seed, use_real_time=use_real_time)
        self.sbox_list = [SBOX[i] for i in range(16)]

    def permute(self, value):
        value &= 0xFFFFFFFF
        out = 0
        for out_index, src_index in enumerate(PBOX_MAP):
            src_bit_pos = 31 - src_index
            dst_bit_pos = 31 - out_index
            bit = (value >> src_bit_pos) & 1
            out |= (bit << dst_bit_pos)
        return out & 0xFFFFFFFF

    def substitute(self, value):
        value &= 0xFFFFFFFF
        out = 0
        for shift in (28, 24, 20, 16, 12, 8, 4, 0):
            nibble = (value >> shift) & 0xF
            out |= (self.sbox_list[nibble] << shift)
        return out & 0xFFFFFFFF

    def gerar(self):
        current_seed = self.seed
        while True:
            x1, x2, x3, x4 = self.produzir_congruencia()
            estado = (current_seed ^ x1) & 0xFFFFFFFF
            estado = self.permute(estado)
            estado = (estado ^ x2) & 0xFFFFFFFF
            estado = self.substitute(estado)
            estado = (estado ^ x3) & 0xFFFFFFFF
            estado = (estado ^ x4) & 0xFFFFFFFF
            self.seed = estado
            current_seed = estado
            if monobit_test_32(estado):
                return estado


# =========================================================
# MERSENNE TWISTER
# =========================================================
class MersenneTwisterWrapper:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def gerar(self):
        return self.rng.getrandbits(32)


# =========================================================
# TESTE QUI-QUADRADO
# =========================================================
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


# =========================================================
# VALIDAÇÃO DE EQUIVALÊNCIA
# =========================================================
def validate_equivalence(seed, count=1000):
    g1 = GNPAOriginal(seed, use_real_time=False)
    g2 = GNPAOptimized(seed, use_real_time=False)

    for i in range(count):
        a = g1.gerar()
        b = g2.gerar()
        if a != b:
            return False, i, a, b
    return True, count, None, None


# =========================================================
# BENCHMARK
# =========================================================
def benchmark_generator(name, generator_factory, sample_sizes, base_seeds):
    results = []

    print(f"\n{'=' * 96}")
    print(f"GERADOR: {name}")
    print(f"{'=' * 96}")

    for size in sample_sizes:
        print(f"\nAmostra: {format_sample_size(size)}")
        print("-" * 96)

        group_rows = []

        for rep, seed in enumerate(base_seeds, start=1):
            env_before = get_system_snapshot()
            gen = generator_factory(seed)

            start = time.perf_counter()
            numbers = [gen.gerar() for _ in range(size)]
            elapsed = time.perf_counter() - start

            chi_sq, p_value = chi_square_test(numbers)
            quality = classify_pvalue(p_value)
            env_after = get_system_snapshot()
            ns_por_numero = (elapsed / size) * 1_000_000_000

            row = {
                "timestamp": env_after["timestamp"],
                "gerador": name,
                "repeticao": rep,
                "seed": seed,
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
                f"Seed: {seed:10d} | "
                f"Tempo: {elapsed:.6f} s | "
                f"ns/número: {ns_por_numero:.2f} | "
                f"Qui²: {chi_sq:.4f} | "
                f"p-valor: {p_value:.6f} | "
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
        print(f"Taxa de aprovação (p >= {ALPHA}): {summary['taxa_aprovacao_percent']:.1f}%")

    return results


# =========================================================
# CSV
# =========================================================
def save_detailed_csv(filepath, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "gerador",
            "repeticao",
            "seed",
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
                r["seed"],
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


# =========================================================
# RELATÓRIO AUTOMÁTICO
# =========================================================
def build_grouped_summary(rows):
    grouped = {}
    for r in rows:
        key = (r["gerador"], r["amostra"])
        grouped.setdefault(key, []).append(r)
    return {k: summarize_group(v) for k, v in grouped.items()}


def winner_by_speed(summary_map, sample_size):
    candidates = []
    for gen in ["GNPA original", "GNPA otimizado", "Mersenne Twister"]:
        s = summary_map[(gen, sample_size)]
        candidates.append((s["ns_por_numero_medio"], gen))
    candidates.sort()
    return candidates[0][1], candidates


def winner_by_pvalue(summary_map, sample_size):
    candidates = []
    for gen in ["GNPA original", "GNPA otimizado", "Mersenne Twister"]:
        s = summary_map[(gen, sample_size)]
        candidates.append((s["p_medio"], gen))
    candidates.sort(reverse=True)
    return candidates[0][1], candidates


def write_automatic_report(filepath, rows, equivalence_ok):
    summary_map = build_grouped_summary(rows)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO AUTOMÁTICO - GNPA ORIGINAL VS GNPA OTIMIZADO VS MT\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Data/hora: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Equivalência GNPA original x otimizado: {'SIM' if equivalence_ok else 'NÃO'}\n\n")

        f.write("1. RESUMO POR AMOSTRA\n")
        f.write("-" * 72 + "\n")
        for size in SAMPLE_SIZES:
            f.write(f"\nAmostra: {format_sample_size(size)}\n")
            for gen in ["GNPA original", "GNPA otimizado", "Mersenne Twister"]:
                s = summary_map[(gen, size)]
                f.write(
                    f"{gen}: tempo médio={s['tempo_medio_s']:.6f}s | "
                    f"ns/número={s['ns_por_numero_medio']:.2f} | "
                    f"Qui² médio={s['qui2_medio']:.4f} | "
                    f"p médio={s['p_medio']:.6f} | "
                    f"aprovação={s['taxa_aprovacao_percent']:.1f}%\n"
                )

        f.write("\n2. VENCEDOR POR VELOCIDADE\n")
        f.write("-" * 72 + "\n")
        for size in SAMPLE_SIZES:
            winner, ranking = winner_by_speed(summary_map, size)
            f.write(f"Amostra {format_sample_size(size)}: {winner}\n")
            for value, gen in ranking:
                f.write(f"  - {gen}: {value:.2f} ns/número\n")

        f.write("\n3. VENCEDOR POR p-VALOR MÉDIO\n")
        f.write("-" * 72 + "\n")
        for size in SAMPLE_SIZES:
            winner, ranking = winner_by_pvalue(summary_map, size)
            f.write(f"Amostra {format_sample_size(size)}: {winner}\n")
            for value, gen in ranking:
                f.write(f"  - {gen}: p médio = {value:.6f}\n")

        f.write("\n4. COMPARAÇÃO DIRETA DO GNPA ORIGINAL COM O OTIMIZADO\n")
        f.write("-" * 72 + "\n")
        for size in SAMPLE_SIZES:
            so = summary_map[("GNPA original", size)]
            sm = summary_map[("GNPA otimizado", size)]

            speedup = (
                so["ns_por_numero_medio"] / sm["ns_por_numero_medio"]
                if sm["ns_por_numero_medio"] > 0 else float("inf")
            )
            delta_p = sm["p_medio"] - so["p_medio"]

            f.write(
                f"Amostra {format_sample_size(size)}: "
                f"speedup = {speedup:.3f}x | "
                f"Δp_médio = {delta_p:+.6f} | "
                f"aprovação original = {so['taxa_aprovacao_percent']:.1f}% | "
                f"aprovação otimizado = {sm['taxa_aprovacao_percent']:.1f}%\n"
            )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Benchmark iniciado...\n")
    print(f"Amostras: {', '.join(format_sample_size(s) for s in SAMPLE_SIZES)}")
    print(f"Repetições por bloco: {REPEATS}")
    print(f"Número de classes (bins): {BINS}")
    print(f"Nível de significância (alpha): {ALPHA}")
    print(f"Modo temporal real do GNPA: {USE_REAL_TIME}")

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

    print("\nValidando equivalência entre GNPA original e GNPA otimizado...")
    ok, where, a, b = validate_equivalence(MASTER_SEED, count=1000)
    if ok:
        print(f"OK: equivalência confirmada nas primeiras {where} saídas.\n")
    else:
        print(f"FALHA: divergência na posição {where}: original={a}, otimizado={b}\n")

    seeds = derive_seeds(MASTER_SEED, REPEATS)

    all_results = []

    all_results.extend(
        benchmark_generator(
            "GNPA original",
            lambda seed: GNPAOriginal(seed, use_real_time=USE_REAL_TIME),
            SAMPLE_SIZES,
            seeds,
        )
    )

    all_results.extend(
        benchmark_generator(
            "GNPA otimizado",
            lambda seed: GNPAOptimized(seed, use_real_time=USE_REAL_TIME),
            SAMPLE_SIZES,
            seeds,
        )
    )

    all_results.extend(
        benchmark_generator(
            "Mersenne Twister",
            lambda seed: MersenneTwisterWrapper(seed),
            SAMPLE_SIZES,
            seeds,
        )
    )

    save_detailed_csv(DETAILS_CSV, all_results)
    save_summary_csv(SUMMARY_CSV, all_results)
    save_environment_csv(ENV_CSV)
    write_automatic_report(REPORT_TXT, all_results, equivalence_ok=ok)

    print("\nArquivos gerados:")
    print(f"- {DETAILS_CSV}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {ENV_CSV}")
    print(f"- {REPORT_TXT}")
    print("\nConcluído.")