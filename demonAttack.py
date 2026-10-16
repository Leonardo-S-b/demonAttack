import argparse
import copy
import csv
import math
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
PENALIDADE_VIDA = 100.0
METAS_PONTUACAO = ((200, 50), (500, 100), (1000, 200))

EXPERIMENTOS = {
    "A": (0.01, 0.05),
    "B": (0.10, 0.20),
    "C": (0.40, 0.50),
}

MODOS_DISPONIVEIS = [1, 3, 5, 7]
DIFICULDADES_DISPONIVEIS = [0, 1]


def calcular_bonus_metas(pontuacao):
    """Soma uma unica vez cada bonus cuja meta foi atingida na partida."""
    return sum(bonus for meta, bonus in METAS_PONTUACAO if pontuacao >= meta)


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
    diagnostico=False,
    penalidade_vida=PENALIDADE_VIDA,
    retornar_detalhes=False,
    usar_bonus_metas=True,
):
    """
    Executa um episodio no Demon Attack e retorna o fitness do agente.

    O ambiente nao informa diretamente a quantidade de inimigos eliminados.
    Fitness = pontos do jogo + bonus das metas - penalidade por vidas perdidas.
    agente=None usa a politica aleatoria com a mesma regra de avaliacao.
    """
    if not math.isfinite(penalidade_vida) or penalidade_vida < 0:
        raise ValueError("A penalidade por vida deve ser finita e nao negativa.")
    ambiente = criar_ambiente(
        render_mode=render_mode,
        modo=modo,
        dificuldade=dificuldade,
    )
    estado, info = ambiente.reset()
    pontuacao = 0.0
    vidas_anteriores = info["lives"]
    vidas_perdidas = 0
    passos_executados = 0
    terminado = truncado = False

    for _ in range(max_passos):
        acao = ambiente.action_space.sample() if agente is None else agente.agir(estado)
        estado, recompensa, terminado, truncado, info = ambiente.step(acao)
        pontuacao += recompensa
        vidas_atuais = info["lives"]
        # Ganhar vidas nao devolve penalidades; perder a ultima vida tambem conta.
        vidas_perdidas += max(0, vidas_anteriores - vidas_atuais)
        vidas_anteriores = vidas_atuais
        passos_executados += 1

        if terminado or truncado:
            break

    ambiente.close()
    bonus_metas = calcular_bonus_metas(pontuacao) if usar_bonus_metas else 0
    fitness = pontuacao + bonus_metas - penalidade_vida * vidas_perdidas
    if diagnostico:
        motivos = []
        if terminado:
            motivos.append("fim de jogo informado pelo Atari (terminated=True)")
        if truncado:
            motivos.append("limite do ambiente ALE/Gymnasium (truncated=True)")
        if not motivos:
            motivos.append("limite de passos definido por --max-passos")
        print(f"Motivo do encerramento: {'; '.join(motivos)}")
        print(f"Passos executados: {passos_executados} / {max_passos}")
        print(f"Vidas restantes informadas pelo ALE: {info.get('lives', 'indisponivel')}")
        print(f"Frames do episodio: {info.get('episode_frame_number', 'indisponivel')}")
        print(f"Vidas perdidas: {vidas_perdidas}")
        print(f"Bonus por metas: {bonus_metas:.2f}")
        print(f"Penalidade total: {penalidade_vida * vidas_perdidas:.2f}")
    if retornar_detalhes:
        return {
            "fitness": fitness,
            "pontuacao": pontuacao,
            "bonus_metas": bonus_metas,
            "vidas_perdidas": vidas_perdidas,
            "penalidade_vida": penalidade_vida,
        }
    return fitness


def visualizar_agente(
    agente,
    max_passos=MAX_PASSOS_EPISODIO,
    modo=MODO_JOGO,
    dificuldade=DIFICULDADE,
    penalidade_vida=PENALIDADE_VIDA,
    usar_bonus_metas=True,
):
    """
    Executa visualmente o melhor agente em uma janela do Atari.
    O agente continua recebendo a observacao pela RAM.
    """
    resultado = avaliar(
        agente,
        max_passos=max_passos,
        render_mode="human",
        modo=modo,
        dificuldade=dificuldade,
        diagnostico=True,
        penalidade_vida=penalidade_vida,
        retornar_detalhes=True,
        usar_bonus_metas=usar_bonus_metas,
    )
    print(f"Pontuacao obtida na visualizacao: {resultado['pontuacao']:.2f}")
    print(f"Fitness final na visualizacao: {resultado['fitness']:.2f}")
    return resultado["fitness"]


def avaliar_agente_aleatorio(
    max_passos=MAX_PASSOS_EPISODIO,
    modo=MODO_JOGO,
    dificuldade=DIFICULDADE,
    penalidade_vida=PENALIDADE_VIDA,
    retornar_detalhes=False,
    usar_bonus_metas=True,
):
    """Baseline: executa um agente que escolhe acoes aleatorias."""
    return avaliar(
        None, max_passos=max_passos, modo=modo, dificuldade=dificuldade,
        penalidade_vida=penalidade_vida, retornar_detalhes=retornar_detalhes,
        usar_bonus_metas=usar_bonus_metas,
    )


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
    penalidade_vida=PENALIDADE_VIDA,
    renderizar_geracoes=False,
    usar_bonus_metas=True,
):
    populacao = criar_populacao(tamanho_populacao)
    historico = []
    melhor_agente = None
    melhor_fitness = float("-inf")
    peso_fase = calcular_peso_fase(modo, dificuldade)
    melhor_fitness_ponderado = float("-inf")

    for geracao in range(num_geracoes):
        for agente in populacao:
            resultado = avaliar(
                agente,
                max_passos=max_passos,
                modo=modo,
                dificuldade=dificuldade,
                penalidade_vida=penalidade_vida,
                retornar_detalhes=True,
                usar_bonus_metas=usar_bonus_metas,
            )
            agente.fitness = resultado["fitness"]
            agente.pontuacao = resultado["pontuacao"]
            agente.bonus_metas = resultado["bonus_metas"]
            agente.usar_bonus_metas = usar_bonus_metas
            agente.vidas_perdidas = resultado["vidas_perdidas"]
            agente.penalidade_vida = penalidade_vida

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
            "penalidade_vida": penalidade_vida,
            "pontuacao_melhor_individuo": melhor_da_geracao.pontuacao,
            "vidas_perdidas_melhor_individuo": melhor_da_geracao.vidas_perdidas,
            "pontuacao_media": float(np.mean([a.pontuacao for a in populacao])),
            "bonus_metas_ativo": usar_bonus_metas,
            "bonus_metas_melhor_individuo": melhor_da_geracao.bonus_metas,
            "bonus_metas_medio": float(np.mean([a.bonus_metas for a in populacao])),
            "vidas_perdidas_media": float(np.mean([a.vidas_perdidas for a in populacao])),
        })

        print(
            f"Experimento {nome} | "
            f"Modo {modo} | "
            f"Dificuldade {dificuldade} | "
            f"Geracao {geracao:03d} | "
            f"Melhor fitness: {melhor:.2f} | "
            f"Pontos: {melhor_da_geracao.pontuacao:.2f} | "
            f"Bonus metas: {melhor_da_geracao.bonus_metas:.2f} | "
            f"Vidas perdidas: {melhor_da_geracao.vidas_perdidas} | "
            f"Ponderado: {melhor_ponderado:.2f} | "
            f"Media: {media:.2f} | "
            f"Pior: {pior:.2f}"
        )

        if renderizar_geracoes:
            print(
                f"Visualizando o melhor da geracao {geracao:03d} "
                f"do experimento {nome} (partida demonstrativa)."
            )
            # A demonstracao nao substitui o fitness usado na selecao ou no CSV.
            visualizar_agente(
                melhor_da_geracao,
                max_passos=max_passos,
                modo=modo,
                dificuldade=dificuldade,
                penalidade_vida=penalidade_vida,
                usar_bonus_metas=usar_bonus_metas,
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
        "penalidade_vida",
        "pontuacao_melhor_individuo",
        "vidas_perdidas_melhor_individuo",
        "pontuacao_media",
        "bonus_metas_ativo",
        "bonus_metas_melhor_individuo",
        "bonus_metas_medio",
        "vidas_perdidas_media",
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
    plt.ylabel("Fitness (pontos + bonus das metas - penalidades)")
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
    penalidade_vida=PENALIDADE_VIDA,
    usar_bonus_metas=True,
):
    caminho_txt.parent.mkdir(parents=True, exist_ok=True)
    media_baseline = float(np.mean([r["fitness"] for r in valores_baseline]))
    melhor_baseline = max(r["fitness"] for r in valores_baseline)

    with caminho_txt.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Comparacao com agente aleatorio\n")
        arquivo.write(f"Penalidade por vida (ambos os agentes): {penalidade_vida:.2f}\n")
        arquivo.write("Fitness = pontos + bonus das metas - penalidade por vida * vidas perdidas\n")
        arquivo.write(f"Metas (pontuacao, bonus): {METAS_PONTUACAO if usar_bonus_metas else ()}\n")
        arquivo.write(f"Episodios do baseline: {len(valores_baseline)}\n")
        arquivo.write(f"Pontuacao media aleatoria: {np.mean([r['pontuacao'] for r in valores_baseline]):.2f}\n")
        arquivo.write(f"Bonus medio por metas aleatorio: {np.mean([r['bonus_metas'] for r in valores_baseline]):.2f}\n")
        arquivo.write(f"Media de vidas perdidas aleatoria: {np.mean([r['vidas_perdidas'] for r in valores_baseline]):.2f}\n")
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
        arquivo.write(f"Pontuacao do jogo: {melhor_info['pontuacao']:.2f}\n")
        arquivo.write(f"Bonus por metas: {melhor_info['bonus_metas']:.2f}\n")
        arquivo.write(f"Metas (pontuacao, bonus): {METAS_PONTUACAO if melhor_info['usar_bonus_metas'] else ()}\n")
        arquivo.write(f"Vidas perdidas: {melhor_info['vidas_perdidas']}\n")
        arquivo.write(f"Penalidade por vida: {melhor_info['penalidade_vida']:.2f}\n")
        arquivo.write(f"Fitness final (antes do peso da fase): {melhor_info['fitness_bruto']:.2f}\n")
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
    parser.add_argument(
        "--sem-bonus-metas", action="store_true",
        help="Desativa os bonus por metas de pontuacao (ativos por padrao).",
    )
    parser.add_argument(
        "--penalidade-vida", type=float, default=PENALIDADE_VIDA,
        help="Pontos descontados do fitness por vida perdida (padrao: 100; 0 desativa).",
    )
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
        "--renderizar-geracoes", action="store_true",
        help="Mostra uma partida do melhor individuo ao final de cada geracao.",
    )
    parser.add_argument(
        "--visualizar-arquivo",
        default=None,
        help="Carrega um agente salvo em .pkl e mostra a partida em uma janela.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.penalidade_vida) or args.penalidade_vida < 0:
        parser.error("--penalidade-vida deve ser um numero finito maior ou igual a zero.")
    return args


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
            penalidade_vida=args.penalidade_vida,
            usar_bonus_metas=not args.sem_bonus_metas,
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
            penalidade_vida=args.penalidade_vida,
            retornar_detalhes=True,
            usar_bonus_metas=not args.sem_bonus_metas,
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
                penalidade_vida=args.penalidade_vida,
                renderizar_geracoes=args.renderizar_geracoes,
                usar_bonus_metas=not args.sem_bonus_metas,
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
                    "pontuacao": melhor_agente.pontuacao,
                    "bonus_metas": melhor_agente.bonus_metas,
                    "usar_bonus_metas": not args.sem_bonus_metas,
                    "vidas_perdidas": melhor_agente.vidas_perdidas,
                    "penalidade_vida": args.penalidade_vida,
                }

    salvar_csv(historico_completo, saida / "fitness.csv")
    gerar_grafico(historico_completo, saida / "fitness.png")
    salvar_baseline(
        valores_baseline,
        melhor_fitness_geral,
        melhor_fitness_ponderado_geral,
        saida / "baseline.txt",
        penalidade_vida=args.penalidade_vida,
        usar_bonus_metas=not args.sem_bonus_metas,
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
            modo=melhor_info_geral["modo"],
            dificuldade=melhor_info_geral["dificuldade"],
            penalidade_vida=args.penalidade_vida,
            usar_bonus_metas=not args.sem_bonus_metas,
        )


if __name__ == "__main__":
    main()
