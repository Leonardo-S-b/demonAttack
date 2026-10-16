import contextlib
import csv
import io
import pickle
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

import demonAttack as projeto


def ambiente_com_metas():
    ambiente = Mock()
    estado = np.zeros(128, dtype=np.uint8)
    ambiente.reset.return_value = (estado, {"lives": 4})
    ambiente.action_space.sample.return_value = 0
    # Continua na partida apos atingir duas metas; nao deve repetir bonus.
    ambiente.step.side_effect = [
        (estado, recompensa, i == 4, False, {"lives": vidas})
        for i, (recompensa, vidas) in enumerate(
            [(200, 4), (0, 4), (300, 3), (0, 2), (180, 0)]
        )
    ]
    return ambiente


def caminho_em_memoria(buffer):
    caminho = MagicMock()
    caminho.open.return_value.__enter__.return_value = buffer
    return caminho


class TestBonusMetas(unittest.TestCase):
    def test_limites_e_bonus_acumulados(self):
        for pontos, bonus in [(0, 0), (199, 0), (200, 50), (499, 50),
                              (500, 150), (999, 150), (1000, 350), (2000, 350)]:
            with self.subTest(pontos=pontos):
                self.assertEqual(projeto.calcular_bonus_metas(pontos), bonus)

    def test_bonus_uma_vez_por_meta_e_sem_acumular_entre_partidas(self):
        agente = Mock()
        for _ in range(2):
            with patch.object(projeto, 'criar_ambiente', return_value=ambiente_com_metas()):
                resultado = projeto.avaliar(agente, retornar_detalhes=True)
            self.assertEqual(resultado['pontuacao'], 680)
            self.assertEqual(resultado['bonus_metas'], 150)
            self.assertEqual(resultado['vidas_perdidas'], 4)
            self.assertEqual(resultado['fitness'], 430)

    def test_baseline_recebe_os_mesmos_bonus(self):
        with patch.object(projeto, 'criar_ambiente', return_value=ambiente_com_metas()):
            resultado = projeto.avaliar_agente_aleatorio(retornar_detalhes=True)
        self.assertEqual(resultado['fitness'], 430)
        self.assertEqual(resultado['bonus_metas'], 150)

    def test_desativar_metas_preserva_penalidade(self):
        with patch.object(projeto, 'criar_ambiente', return_value=ambiente_com_metas()):
            resultado = projeto.avaliar(Mock(), retornar_detalhes=True, usar_bonus_metas=False)
        self.assertEqual(resultado['bonus_metas'], 0)
        self.assertEqual(resultado['fitness'], 280)

    def test_visualizacao_mostra_bonus_e_fitness(self):
        saida = io.StringIO()
        with patch.object(projeto, 'criar_ambiente', return_value=ambiente_com_metas()), \
             contextlib.redirect_stdout(saida):
            projeto.visualizar_agente(Mock())
        self.assertIn('Bonus por metas: 150.00', saida.getvalue())
        self.assertIn('Pontuacao obtida na visualizacao: 680.00', saida.getvalue())
        self.assertIn('Fitness final na visualizacao: 430.00', saida.getvalue())

    def test_cli_visualizacao_desativa_metas(self):
        with patch('sys.argv', ['main.py', '--visualizar-arquivo', 'teste.pkl', '--sem-bonus-metas']), \
             patch.object(projeto, 'carregar_agente'), \
             patch.object(projeto, 'visualizar_agente') as visualizar:
            projeto.main()
        self.assertFalse(visualizar.call_args.kwargs['usar_bonus_metas'])

    def test_fluxo_e_artefatos_em_memoria(self):
        # Serializadores reais, mas destinos em memoria: sem treino Atari ou arquivos.
        for desativar in (False, True):
            with self.subTest(desativar=desativar):
                csv_buffer, baseline_buffer, resumo_buffer = (io.StringIO() for _ in range(3))
                pickle_buffer, png_buffer = io.BytesIO(), io.BytesIO()
                originais = {nome: getattr(projeto, nome) for nome in
                             ('salvar_csv', 'salvar_baseline', 'salvar_resumo_melhor', 'salvar_melhor_agente')}
                argv = ['main.py', '--experimentos', 'todos', '--populacao', '4',
                        '--geracoes', '2', '--baseline-episodios', '1', '--renderizar-geracoes',
                        '--renderizar-melhor'] + (['--sem-bonus-metas'] if desativar else [])
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch('sys.argv', argv))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    stack.enter_context(patch.object(projeto, 'criar_ambiente', side_effect=lambda **kw: ambiente_com_metas()))
                    visualizar = stack.enter_context(patch.object(projeto, 'visualizar_agente'))
                    stack.enter_context(patch.object(projeto, 'salvar_csv', side_effect=lambda h, p: originais['salvar_csv'](h, caminho_em_memoria(csv_buffer))))
                    stack.enter_context(patch.object(projeto, 'salvar_baseline', side_effect=lambda v, m, mp, p, **kw: originais['salvar_baseline'](v, m, mp, caminho_em_memoria(baseline_buffer), **kw)))
                    stack.enter_context(patch.object(projeto, 'salvar_resumo_melhor', side_effect=lambda i, p: originais['salvar_resumo_melhor'](i, caminho_em_memoria(resumo_buffer))))
                    stack.enter_context(patch.object(projeto, 'salvar_melhor_agente', side_effect=lambda a, p: originais['salvar_melhor_agente'](a, caminho_em_memoria(pickle_buffer))))
                    stack.enter_context(patch.object(projeto.Path, 'mkdir'))
                    stack.enter_context(patch.object(projeto.plt, 'savefig', side_effect=lambda p: projeto.plt.gcf().savefig(png_buffer, format='png')))
                    projeto.main()
                bonus, fitness = (0, 280) if desativar else (150, 430)
                linhas = list(csv.DictReader(io.StringIO(csv_buffer.getvalue())))
                self.assertEqual(len(linhas), 6)
                for linha in linhas:
                    self.assertEqual(float(linha['melhor']), fitness)
                    self.assertEqual(float(linha['bonus_metas_melhor_individuo']), bonus)
                    self.assertEqual(float(linha['bonus_metas_medio']), bonus)
                    self.assertEqual(linha['bonus_metas_ativo'], str(not desativar))
                self.assertIn(f'Bonus por metas: {bonus:.2f}', resumo_buffer.getvalue())
                self.assertIn(f'Bonus medio por metas aleatorio: {bonus:.2f}', baseline_buffer.getvalue())
                self.assertIn(f'Fitness medio aleatorio: {fitness:.2f}', baseline_buffer.getvalue())
                agente = pickle.loads(pickle_buffer.getvalue())
                self.assertEqual(agente.bonus_metas, bonus)
                self.assertEqual(agente.fitness, fitness)
                self.assertTrue(png_buffer.getvalue().startswith(b'\x89PNG'))
                self.assertEqual(visualizar.call_count, 7)
                self.assertTrue(all(c.kwargs['usar_bonus_metas'] == (not desativar)
                                    for c in visualizar.call_args_list))


if __name__ == '__main__':
    unittest.main()
