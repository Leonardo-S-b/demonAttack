import argparse
import copy
import csv
import pickle
import random
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

gym.register_envs(ale_py)

# ---------------------------------------------------------
# CONFIGURACOES
# ---------------------------------------------------------

ENV_ID = "ALE/DemonAttack-v5"

TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ENTRADAS = 128
NUM_OCULTOS = 64
NUM_ACOES = 6
MAX_PASSOS_EPISODIO = 5000
MODO_JOGO = 1
DIFICULDADE = 0

TAXA_MUTACAO = 0.10
INTENSIDADE_MUTACAO = 0.20
NUM_ELITE = 3

EXPERIMENTOS = {
    "A": (0.01, 0.05),
    "B": (0.10, 0.20),
    "C": (0.40, 0.50),
}

MODOS_DISPONIVEIS = [1, 3, 5, 7]
DIFICULDADES_DISPONIVEIS = [0, 1]


def calcular_peso_fase(modo, dificuldade):
    """
    Peso usado para comparar resultados de configuracoes diferentes.
    Modos maiores e dificuldade 1 recebem mais peso porque tendem a ser
    configuracoes mais exigentes do Demon Attack.
    """
    return 1.0 + (modo * 0.15) + (dificuldade * 0.50)


def criar_ambiente(render_mode=None, modo=MODO_JOGO, dificuldade=DIFICULDADE):
    return gym.make(
        ENV_ID,
        obs_type="ram",
        mode=modo,
        difficulty=dificuldade,
        render_mode=render_mode,
    )


class Agente:
    def __init__(self):
        # Cada individuo comeca com parametros diferentes.
        self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
        self.b1 = np.random.randn(NUM_OCULTOS)
        self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
        self.b2 = np.random.randn(NUM_ACOES)
        self.fitness = 0.0

    def agir(self, estado):
        # RAM: valores inteiros entre 0 e 255.
        estado = estado.astype(np.float32) / 255.0

        oculto = np.tanh(estado @ self.W1 + self.b1)
        saida = oculto @ self.W2 + self.b2

        # A maior saida define a acao escolhida.
        return int(np.argmax(saida))


def avaliar(
    agente,
    max_passos=MAX_PASSOS_EPISODIO,
    render_mode=None,
    modo=MODO_JOGO,
    dificuldade=DIFICULDADE,
):
    """
    Executa um episodio no Demon Attack e retorna o fitness do agente.

    O ambiente nao informa diretamente a quantidade de inimigos eliminados.
    Por isso, o fitness e a soma das recompensas/pontos obtidos no episodio.
    """
    ambiente = criar_ambiente(
        render_mode=render_mode,
        modo=modo,
        dificuldade=dificuldade,
    )
    estado, _ = ambiente.reset()
    fitness = 0.0

    for _ in range(max_passos):
        acao = agente.agir(estado)
        estado, recompensa, terminado, truncado, _ = ambiente.step(acao)
        fitness += recompensa

        if terminado or truncado:
            break

    ambiente.close()
    return fitness


def visualizar_agente(
    agente,
    max_passos=MAX_PASSOS_EPISODIO,
    modo=MODO_JOGO,
    dificuldade=DIFICULDADE,
):
    """
    Executa visualmente o melhor agente em uma janela do Atari.
    O agente continua recebendo a observacao pela RAM.
    """
    recompensa_total = avaliar(
        agente,
        max_passos=max_passos,
        render_mode="human",
        modo=modo,
        dificuldade=dificuldade,
    )
    print(f"Pontuacao obtida na visualizacao: {recompensa_total:.2f}")
    return recompensa_total


def avaliar_agente_aleatorio(
    max_passos=MAX_PASSOS_EPISODIO,
    modo=MODO_JOGO,
    dificuldade=DIFICULDADE,
):
    """Baseline: executa um agente que escolhe acoes aleatorias."""
    ambiente = criar_ambiente(modo=modo, dificuldade=dificuldade)
    estado, _ = ambiente.reset()
    fitness = 0.0

    for _ in range(max_passos):
        acao = ambiente.action_space.sample()
        estado, recompensa, terminado, truncado, _ = ambiente.step(acao)
        fitness += recompensa

        if terminado or truncado:
            break

    ambiente.close()
    return fitness


def criar_populacao(tamanho=TAMANHO_POPULACAO):
    """
    Cria varios agentes independentes.
    Cada agente possui seus proprios pesos e bias.
    """
    return [Agente() for _ in range(tamanho)]


def selecionar(populacao):
    """
    Selecao por ranking.
    Os individuos sao ordenados pelo fitness, e a metade superior pode gerar filhos.
    """
    ordenados = sorted(populacao, key=lambda agente: agente.fitness, reverse=True)
    quantidade_pais = max(1, len(populacao) // 2)
    return ordenados[:quantidade_pais]


def mutar(agente, taxa_mutacao=TAXA_MUTACAO, intensidade_mutacao=INTENSIDADE_MUTACAO):
    """
    Altera aleatoriamente parte dos pesos e bias do agente.
    A taxa define quantos parametros tendem a mudar; a intensidade define o tamanho
    medio da alteracao.
    """
    for parametro in (agente.W1, agente.b1, agente.W2, agente.b2):
        mascara = np.random.random(parametro.shape) < taxa_mutacao
        ruido = np.random.randn(*parametro.shape) * intensidade_mutacao
        parametro += mascara * ruido

    return agente


def nova_geracao(
    populacao,
    taxa_mutacao=TAXA_MUTACAO,
    intensidade_mutacao=INTENSIDADE_MUTACAO,
    num_elite=NUM_ELITE,
):
    """
    Combina selecao, elitismo, reproducao e mutacao.
    O filho e uma copia independente de um pai, seguida de mutacao.
    """
    pais = selecionar(populacao)
    ordenados = sorted(populacao, key=lambda agente: agente.fitness, reverse=True)

    quantidade_elite = min(num_elite, len(populacao))
    nova_populacao = [copy.deepcopy(agente) for agente in ordenados[:quantidade_elite]]

    while len(nova_populacao) < len(populacao):
        pai = random.choice(pais)
        filho = copy.deepcopy(pai)
        filho.fitness = 0.0
        nova_populacao.append(mutar(filho, taxa_mutacao, intensidade_mutacao))

    return nova_populacao


def executar_experimento(
    nome,
    taxa_mutacao,
    intensidade_mutacao,
    tamanho_populacao,
    num_geracoes,
    max_passos,
    modo,
    dificuldade,
):
    populacao = criar_populacao(tamanho_populacao)
    historico = []
    melhor_agente = None
    melhor_fitness = float("-inf")
    peso_fase = calcular_peso_fase(modo, dificuldade)
    melhor_fitness_ponderado = float("-inf")

    for geracao in range(num_geracoes):
        for agente in populacao:
            agente.fitness = avaliar(
                agente,
                max_passos=max_passos,
                modo=modo,
                dificuldade=dificuldade,
            )

        valores = [agente.fitness for agente in populacao]
        melhor = max(valores)
        media = float(np.mean(valores))
        pior = min(valores)

        melhor_da_geracao = max(populacao, key=lambda agente: agente.fitness)
        melhor_ponderado = melhor * peso_fase
        media_ponderada = media * peso_fase

        if melhor_ponderado > melhor_fitness_ponderado:
            melhor_fitness = melhor_da_geracao.fitness
            melhor_fitness_ponderado = melhor_ponderado
            melhor_agente = copy.deepcopy(melhor_da_geracao)

        historico.append({
            "experimento": nome,
            "geracao": geracao,
            "taxa_mutacao": taxa_mutacao,
            "intensidade_mutacao": intensidade_mutacao,
            "modo": modo,
            "dificuldade": dificuldade,
            "peso_fase": peso_fase,
            "melhor": melhor,
            "melhor_ponderado": melhor_ponderado,
            "media": media,
            "media_ponderada": media_ponderada,
            "pior": pior,
        })

        print(
            f"Experimento {nome} | "
            f"Modo {modo} | "
            f"Dificuldade {dificuldade} | "
            f"Geracao {geracao:03d} | "
            f"Melhor: {melhor:.2f} | "
            f"Ponderado: {melhor_ponderado:.2f} | "
            f"Media: {media:.2f} | "
            f"Pior: {pior:.2f}"
        )

        populacao = nova_geracao(
            populacao,
            taxa_mutacao=taxa_mutacao,
            intensidade_mutacao=intensidade_mutacao,
        )

    return historico, melhor_agente, melhor_fitness, melhor_fitness_ponderado


def salvar_csv(historico, caminho_csv):
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "experimento",
        "geracao",
        "taxa_mutacao",
        "intensidade_mutacao",
        "modo",
        "dificuldade",
        "peso_fase",
        "melhor",
        "melhor_ponderado",
        "media",
        "media_ponderada",
        "pior",
    ]

    with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(historico)


def gerar_grafico(historico, caminho_png):
    caminho_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    nomes = sorted({linha["experimento"] for linha in historico})

    for nome in nomes:
        linhas = [linha for linha in historico if linha["experimento"] == nome]
        geracoes = [linha["geracao"] for linha in linhas]
        melhores = [linha["melhor"] for linha in linhas]
        medias = [linha["media"] for linha in linhas]

        plt.plot(geracoes, melhores, marker="o", label=f"{nome} - melhor")
        plt.plot(geracoes, medias, linestyle="--", label=f"{nome} - media")

    plt.title("Evolucao do fitness no Demon Attack")
    plt.xlabel("Geracao")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_png)
    plt.close()


def salvar_baseline(
    valores_baseline,
    melhor_evoluido,
    melhor_evoluido_ponderado,
    caminho_txt,
):
    caminho_txt.parent.mkdir(parents=True, exist_ok=True)
    media_baseline = float(np.mean(valores_baseline))
    melhor_baseline = max(valores_baseline)

    with caminho_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Comparacao com agente aleatorio\n")
        arquivo.write(f"Episodios do baseline: {len(valores_baseline)}\n")
        arquivo.write(f"Fitness medio aleatorio: {media_baseline:.2f}\n")
        arquivo.write(f"Melhor fitness aleatorio: {melhor_baseline:.2f}\n")
        arquivo.write(f"Melhor fitness evoluido: {melhor_evoluido:.2f}\n")
        arquivo.write(
            f"Melhor fitness evoluido ponderado: {melhor_evoluido_ponderado:.2f}\n"
        )


def salvar_resumo_melhor(melhor_info, caminho_txt):
    caminho_txt.parent.mkdir(parents=True, exist_ok=True)

    with caminho_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Melhor agente geral\n")
        arquivo.write(f"Experimento: {melhor_info['experimento']}\n")
        arquivo.write(f"Modo: {melhor_info['modo']}\n")
        arquivo.write(f"Dificuldade: {melhor_info['dificuldade']}\n")
        arquivo.write(f"Peso da fase: {melhor_info['peso_fase']:.2f}\n")
        arquivo.write(f"Fitness bruto: {melhor_info['fitness_bruto']:.2f}\n")
        arquivo.write(
            f"Fitness ponderado: {melhor_info['fitness_ponderado']:.2f}\n"
        )


def salvar_melhor_agente(agente, caminho_pkl):
    caminho_pkl.parent.mkdir(parents=True, exist_ok=True)

    with caminho_pkl.open("wb") as arquivo:
        pickle.dump(agente, arquivo)


def carregar_agente(caminho_pkl):
    if not caminho_pkl.exists():
        raise FileNotFoundError(
            f"Nao encontrei o agente salvo em '{caminho_pkl}'. "
            "Treine primeiro com: python demonAttack.py"
        )

    with caminho_pkl.open("rb") as arquivo:
        return pickle.load(arquivo)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neuroevolucao aplicada ao Atari Demon Attack."
    )
    parser.add_argument("--experimentos", default="B", choices=["A", "B", "C", "todos"])
    parser.add_argument("--populacao", type=int, default=TAMANHO_POPULACAO)
    parser.add_argument("--geracoes", type=int, default=NUM_GERACOES)
    parser.add_argument("--max-passos", type=int, default=MAX_PASSOS_EPISODIO)
    parser.add_argument("--baseline-episodios", type=int, default=5)
    parser.add_argument("--modo", type=int, default=MODO_JOGO, choices=MODOS_DISPONIVEIS)
    parser.add_argument(
        "--dificuldade",
        type=int,
        default=DIFICULDADE,
        choices=DIFICULDADES_DISPONIVEIS,
    )
    parser.add_argument(
        "--variacoes-fase",
        action="store_true",
        help="Executa todos os modos disponiveis nas dificuldades 0 e 1.",
    )
    parser.add_argument("--saida", default="resultados")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--renderizar-melhor", action="store_true")
    parser.add_argument(
        "--visualizar-arquivo",
        default=None,
        help="Carrega um agente salvo em .pkl e mostra a partida em uma janela.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    saida = Path(args.saida)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.visualizar_arquivo is not None:
        agente = carregar_agente(Path(args.visualizar_arquivo))
        visualizar_agente(
            agente,
            max_passos=args.max_passos,
            modo=args.modo,
            dificuldade=args.dificuldade,
        )
        return

    nomes_experimentos = (
        list(EXPERIMENTOS)
        if args.experimentos == "todos"
        else [args.experimentos]
    )
    fases = (
        [
            (modo, dificuldade)
            for modo in MODOS_DISPONIVEIS
            for dificuldade in DIFICULDADES_DISPONIVEIS
        ]
        if args.variacoes_fase
        else [(args.modo, args.dificuldade)]
    )

    print("Avaliando agente aleatorio de referencia...")
    valores_baseline = [
        avaliar_agente_aleatorio(
            max_passos=args.max_passos,
            modo=args.modo,
            dificuldade=args.dificuldade,
        )
        for _ in range(args.baseline_episodios)
    ]

    historico_completo = []
    melhor_agente_geral = None
    melhor_fitness_geral = float("-inf")
    melhor_fitness_ponderado_geral = float("-inf")
    melhor_info_geral = None

    for modo, dificuldade in fases:
        for nome in nomes_experimentos:
            taxa_mutacao, intensidade_mutacao = EXPERIMENTOS[nome]
            nome_execucao = f"{nome}-m{modo}-d{dificuldade}"
            (
                historico,
                melhor_agente,
                melhor_fitness,
                melhor_fitness_ponderado,
            ) = executar_experimento(
                nome=nome_execucao,
                taxa_mutacao=taxa_mutacao,
                intensidade_mutacao=intensidade_mutacao,
                tamanho_populacao=args.populacao,
                num_geracoes=args.geracoes,
                max_passos=args.max_passos,
                modo=modo,
                dificuldade=dificuldade,
            )
            historico_completo.extend(historico)

            if melhor_fitness_ponderado > melhor_fitness_ponderado_geral:
                melhor_fitness_geral = melhor_fitness
                melhor_fitness_ponderado_geral = melhor_fitness_ponderado
                melhor_agente_geral = melhor_agente
                melhor_info_geral = {
                    "experimento": nome_execucao,
                    "modo": modo,
                    "dificuldade": dificuldade,
                    "peso_fase": calcular_peso_fase(modo, dificuldade),
                    "fitness_bruto": melhor_fitness,
                    "fitness_ponderado": melhor_fitness_ponderado,
                }

    salvar_csv(historico_completo, saida / "fitness.csv")
    gerar_grafico(historico_completo, saida / "fitness.png")
    salvar_baseline(
        valores_baseline,
        melhor_fitness_geral,
        melhor_fitness_ponderado_geral,
        saida / "baseline.txt",
    )
    salvar_resumo_melhor(melhor_info_geral, saida / "melhor_agente.txt")
    salvar_melhor_agente(melhor_agente_geral, saida / "melhor_agente.pkl")

    print(f"CSV salvo em: {saida / 'fitness.csv'}")
    print(f"Grafico salvo em: {saida / 'fitness.png'}")
    print(f"Baseline salvo em: {saida / 'baseline.txt'}")
    print(f"Resumo do melhor agente salvo em: {saida / 'melhor_agente.txt'}")
    print(f"Melhor agente salvo em: {saida / 'melhor_agente.pkl'}")

    if args.renderizar_melhor and melhor_agente_geral is not None:
        visualizar_agente(
            melhor_agente_geral,
            max_passos=args.max_passos,
            modo=args.modo,
            dificuldade=args.dificuldade,
        )


if __name__ == "__main__":
    main()
