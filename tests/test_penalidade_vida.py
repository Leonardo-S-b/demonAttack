import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import demonAttack as projeto


def ambiente_simulado(vidas_iniciais=4, vidas=(3, 0), terminado=True, truncado=False):
    ambiente = Mock()
    estado = np.zeros(128, dtype=np.uint8)
    ambiente.reset.return_value = (estado, {"lives": vidas_iniciais})
    ambiente.action_space.sample.return_value = 0
    ambiente.step.side_effect = [
        (estado, 10.0, terminado and i == len(vidas) - 1,
         truncado and i == len(vidas) - 1,
         {"lives": vida, "episode_frame_number": (i + 1) * 4})
        for i, vida in enumerate(vidas)
    ]
    return ambiente


class TestPenalidadeVida(unittest.TestCase):
    def avaliar_simulado(self, vidas_iniciais, vidas, **kwargs):
        ambiente = ambiente_simulado(vidas_iniciais, vidas)
        with patch.object(projeto, "criar_ambiente", return_value=ambiente):
            resultado = projeto.avaliar(Mock(), max_passos=len(vidas),
                                       retornar_detalhes=True, **kwargs)
        ambiente.close.assert_called_once()
        return resultado

    def test_cada_vida_inclusive_ultima_e_fitness_negativo(self):
        resultado = self.avaliar_simulado(4, (3, 3, 0))
        self.assertEqual(resultado["vidas_perdidas"], 4)
        self.assertEqual(resultado["pontuacao"], 30)
        self.assertEqual(resultado["fitness"], -370)

    def test_vidas_extras_nao_devolvem_penalidades(self):
        resultado = self.avaliar_simulado(1, (4, 3, 4, 3))
        self.assertEqual(resultado["vidas_perdidas"], 2)
        self.assertEqual(resultado["fitness"], -160)

    def test_zero_sem_metas_reproduz_fitness_antigo(self):
        resultado = self.avaliar_simulado(4, (3, 0), penalidade_vida=0, usar_bonus_metas=False)
        self.assertEqual(resultado["fitness"], resultado["pontuacao"])

    def test_penalidade_personalizada(self):
        resultado = self.avaliar_simulado(4, (3,), penalidade_vida=250)
        self.assertEqual(resultado["fitness"], -240)

    def test_limites_nao_sao_perdas_de_vida(self):
        for truncado in (False, True):
            with self.subTest(truncado=truncado):
                ambiente = ambiente_simulado(4, (4,), terminado=False, truncado=truncado)
                with patch.object(projeto, "criar_ambiente", return_value=ambiente):
                    resultado = projeto.avaliar(Mock(), max_passos=1, retornar_detalhes=True)
                self.assertEqual(resultado["vidas_perdidas"], 0)
                self.assertEqual(resultado["fitness"], 10)

    def test_baseline_usa_mesma_regra(self):
        ambiente = ambiente_simulado()
        with patch.object(projeto, "criar_ambiente", return_value=ambiente):
            resultado = projeto.avaliar_agente_aleatorio(retornar_detalhes=True)
        self.assertEqual(resultado["fitness"], -380)
        self.assertEqual(ambiente.action_space.sample.call_count, 2)

    def test_visualizacao_separa_pontos_e_fitness(self):
        saida = io.StringIO()
        with patch.object(projeto, "criar_ambiente", return_value=ambiente_simulado()), \
             contextlib.redirect_stdout(saida):
            projeto.visualizar_agente(Mock(), penalidade_vida=50)
        self.assertIn("Pontuacao obtida na visualizacao: 20.00", saida.getvalue())
        self.assertIn("Fitness final na visualizacao: -180.00", saida.getvalue())
        self.assertIn("Vidas perdidas: 4", saida.getvalue())

    def test_cli_rejeita_valores_invalidos(self):
        for valor in ("-1", "nan", "inf"):
            with self.subTest(valor=valor), \
                 patch("sys.argv", ["main.py", "--penalidade-vida", valor]), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as erro:
                    projeto.parse_args()
                self.assertEqual(erro.exception.code, 2)

    def test_fluxo_completo_com_ambiente_simulado(self):
        # Valida CSV, grafico, baseline e pickle sem executar partidas do Atari.
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as pasta:
            argv = ["main.py", "--experimentos", "todos", "--populacao", "4",
                    "--geracoes", "2", "--baseline-episodios", "1",
                    "--penalidade-vida", "50", "--saida", pasta]
            with patch("sys.argv", argv), \
                 patch.object(projeto, "criar_ambiente", side_effect=lambda **kw: ambiente_simulado()), \
                 contextlib.redirect_stdout(io.StringIO()):
                projeto.main()
            destino = Path(pasta)
            with (destino / "fitness.csv").open() as arquivo:
                linhas = list(csv.DictReader(arquivo))
            self.assertEqual(len(linhas), 6)
            for linha in linhas:
                self.assertEqual(float(linha["melhor"]), -180)
                self.assertEqual(float(linha["pontuacao_melhor_individuo"]), 20)
                self.assertEqual(float(linha["vidas_perdidas_melhor_individuo"]), 4)
                self.assertEqual(float(linha["penalidade_vida"]), 50)
            self.assertGreater((destino / "fitness.png").stat().st_size, 0)
            self.assertIn("Fitness medio aleatorio: -180.00", (destino / "baseline.txt").read_text())
            self.assertIn("Pontuacao do jogo: 20.00", (destino / "melhor_agente.txt").read_text())
            agente = projeto.carregar_agente(destino / "melhor_agente.pkl")
            self.assertEqual(agente.fitness, -180)
            self.assertEqual(agente.penalidade_vida, 50)


if __name__ == "__main__":
    unittest.main()
