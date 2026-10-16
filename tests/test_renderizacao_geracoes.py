import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import demonAttack as projeto


class TestRenderizacaoGeracoes(unittest.TestCase):
    def test_exibe_campeao_de_cada_geracao_sem_alterar_fitness(self):
        populacao = [SimpleNamespace(fitness=0) for _ in range(4)]
        resultados = [dict(fitness=f, pontuacao=f + 100, vidas_perdidas=1, bonus_metas=0)
                      for f in (40, 30, 20, 10, 1, 2, 3, 4)]
        exibidos = []

        def visualizar(agente, **kwargs):
            exibidos.append((populacao.index(agente), agente.fitness, kwargs))
            return 9999  # A demonstracao nao deve entrar no historico/recorde.

        with patch.object(projeto, 'criar_populacao', return_value=populacao), \
             patch.object(projeto, 'avaliar', side_effect=resultados), \
             patch.object(projeto, 'nova_geracao', return_value=populacao), \
             patch.object(projeto, 'visualizar_agente', side_effect=visualizar), \
             contextlib.redirect_stdout(io.StringIO()):
            historico, campeao, fitness, _ = projeto.executar_experimento(
                'B-m3-d1', 0.1, 0.2, 4, 2, 123, 3, 1,
                penalidade_vida=100, renderizar_geracoes=True,
            )
        self.assertEqual([(i, f) for i, f, _ in exibidos], [(0, 40), (3, 4)])
        for _, _, argumentos in exibidos:
            self.assertEqual(argumentos, dict(max_passos=123, modo=3, dificuldade=1, penalidade_vida=100, usar_bonus_metas=True))
        self.assertEqual([linha['melhor'] for linha in historico], [40, 4])
        self.assertEqual(fitness, 40)
        self.assertEqual(campeao.fitness, 40)

    def test_sem_opcao_nao_abre_janela(self):
        resultado = dict(fitness=10, pontuacao=110, vidas_perdidas=1, bonus_metas=0)
        with patch.object(projeto, 'avaliar', return_value=resultado), \
             patch.object(projeto, 'visualizar_agente') as visualizar, \
             contextlib.redirect_stdout(io.StringIO()):
            projeto.executar_experimento('B', 0.1, 0.2, 4, 1, 100, 1, 0)
        visualizar.assert_not_called()

    def test_cli_padrao_e_opcao(self):
        with patch('sys.argv', ['main.py']):
            self.assertFalse(projeto.parse_args().renderizar_geracoes)
        with patch('sys.argv', ['main.py', '--renderizar-geracoes']):
            args = projeto.parse_args()
            self.assertTrue(args.renderizar_geracoes)
            self.assertEqual(args.penalidade_vida, 100)

    def test_main_encaminha_opcao_aos_tres_experimentos(self):
        campeao = SimpleNamespace(pontuacao=110, vidas_perdidas=1, bonus_metas=0)
        # Sem partidas, janelas ou gravacao de arquivos neste teste.
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('sys.argv', ['main.py', '--experimentos', 'todos', '--renderizar-geracoes']))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(projeto, 'avaliar_agente_aleatorio'))
            executar = stack.enter_context(patch.object(projeto, 'executar_experimento', return_value=([], campeao, 10, 11.5)))
            for nome in ('salvar_csv', 'gerar_grafico', 'salvar_baseline',
                         'salvar_resumo_melhor', 'salvar_melhor_agente'):
                stack.enter_context(patch.object(projeto, nome))
            projeto.main()
        self.assertEqual(executar.call_count, 3)
        self.assertEqual([c.kwargs['nome'] for c in executar.call_args_list],
                         ['A-m1-d0', 'B-m1-d0', 'C-m1-d0'])
        self.assertTrue(all(c.kwargs['renderizar_geracoes'] for c in executar.call_args_list))


if __name__ == '__main__':
    unittest.main()
