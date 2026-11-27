# Market Maker Fear Gauge

Na Hyperliquid, tudo o que um market maker faz fica público: cada cotação, cada execução e a posição da carteira. Este projeto usa esses dados para medir a **aversão a risco (γ)** de um market maker real e acompanhar como ela muda ao longo do dia, que é o "fear gauge". Depois, testa onde o modelo de livro-texto deixa de explicar o comportamento dele.

## A ideia

O modelo de Avellaneda-Stoikov, na versão de horizonte longo de Guéant, Lehalle e Fernandez-Tapia, descreve um market maker que não gosta de carregar estoque. Ele prevê duas coisas observáveis:

- **Deslocamento:** com inventário positivo, o market maker baixa as duas cotações para vender logo. O tamanho desse deslocamento por unidade de inventário depende de γ.
- **Spread:** a distância entre compra e venda depende de γ e, principalmente, da velocidade com que as ordens são executadas.

Estimando a volatilidade e a intensidade de execução, dá para inverter as fórmulas e descobrir qual γ explica as cotações observadas. Se as duas fórmulas derem γ muito diferentes, o modelo não descreve esse market maker.

## Validação antes dos dados reais

O notebook `00_validacao_simulada` cria um market maker com γ conhecido e confere se o método recupera esse valor. Resultado: o γ medido pelo deslocamento acerta com erro de 4% a 10%, mas o γ medido pelo spread é instável e, com tick mínimo, chega a ficar 50 vezes acima do verdadeiro. Por isso, o deslocamento é a medida principal e o spread vira teste do modelo.

## Testes de onde o modelo quebra

| Teste | O que o modelo supõe | O que pode acontecer na prática |
|---|---|---|
| Forma do deslocamento | Linear no inventário | Limites de risco: para de cotar um lado perto do limite |
| Markout depois do fill | O preço não anda contra o market maker | Seleção adversa: quem executa sabe algo antes |
| Tick mínimo | Preços contínuos | Spread preso em 1 ou 2 ticks |
| γ vs volatilidade | γ constante | Mais medo quando o mercado fica agitado |
| Intensidade de execução | Queda exponencial com a distância | Queda em lei de potência, assimetria compra/venda |
| Meia-vida do inventário | Inventário volta a zero pelas cotações | Proteção (hedge) em outra corretora |

## Como rodar (Windows)

1. `1_instalar_e_validar.bat` instala as dependências, roda os testes e executa o notebook 00.
2. `2_descobrir_market_maker.bat` ouve os trades de ETH, SOL e HYPE por 30 minutos, ranqueia as carteiras e salva a que mais parece market maker em `data/alvo.json`.
3. `3_gravar_24h.bat` grava cotações, fills, inventário e o livro dessa carteira por 24 horas. O PC precisa ficar ligado.
4. `4_analisar.bat` executa o notebook 01 com a gravação mais recente.

Em outros sistemas: `pip install -e ".[dev]"`, depois `python -m fear_gauge.discover`, `python -m fear_gauge.recorder --horas 24` e os notebooks.

## Estrutura

| Arquivo | Papel |
|---|---|
| `glft.py` | Fórmulas do modelo e inversão de γ com SciPy |
| `simulate.py` | Market maker sintético, com opções de seleção adversa, limite de inventário e tick |
| `estimate.py` | Volatilidade, intensidade de execução (máxima verossimilhança), deslocamento (OLS com erros HAC do statsmodels) e fear gauge |
| `breaks.py` | Testes de quebra do modelo |
| `hl_api.py`, `recorder.py`, `discover.py` | Coleta de dados da Hyperliquid por REST e websocket |
| `dataset.py` | Transforma a gravação em um painel de 1 segundo |

## Limitações

A posição visível na Hyperliquid pode não ser o risco real do market maker, se ele se protege em outra corretora ou divide a operação em várias carteiras. A API não guarda histórico de cotações, então o estudo cobre só o período gravado. E γ aqui é o parâmetro que faz o modelo reproduzir o comportamento observado, não uma medida direta do que o operador sente.

## Resultados

Market maker de HYPE (carteira `0x4e60e3a4...`), cerca de 22 minutos gravados em 26/11/2026, 281 execuções como maker.

| Medida | Resultado |
|---|---|
| γ pelo deslocamento das cotações | 0,06 (intervalo de 0,035 a 0,095) |
| γ pelo spread | sem solução: o spread observado (0,0164 US$) é menor que o mínimo do modelo (0,017 US$) |
| Deslocamento linear no inventário | sim (teste de curvatura com p-valor 0,70) |
| Preço anda contra ele 1 s depois do fill | sim, nos dois lados (p-valor abaixo de 0,001) |
| Intensidade de execução exponencial | não, lei de potência se ajusta melhor |
| γ estável quando a volatilidade muda | não: variou de 0,05 a 0,42, na direção contrária da volatilidade |

Dá para medir uma aversão a risco, mas o modelo de livro-texto explica só uma parte do comportamento. Esse market maker cota mais apertado do que o modelo permite, sofre seleção adversa e parece usar uma regra fixa de deslocamento em vez de ajustar o risco à volatilidade. Com só 22 minutos e um único market maker, os resultados são um primeiro retrato, não uma conclusão definitiva. Detalhes na seção 6 do notebook 01.
