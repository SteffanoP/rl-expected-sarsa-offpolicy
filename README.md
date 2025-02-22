# Expected SARSA Off-Policy

## O que é o SARSA?

SARSA (State-Action-Reward-State-Action) é um algoritmo de aprendizado por reforço para estimar funções de valor-ação. O Expected SARSA é uma variante que, em vez de usar a ação real tomada no próximo estado para atualizar os valores Q, usa o valor esperado considerando todas as possíveis ações no próximo estado.

A atualização do Expected SARSA é dada por:

$$ Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \mathbb{E}[Q(S',A')] - Q(S,A)] $$

onde:

- $\alpha$ é a taxa de aprendizado
- $\gamma$ é o fator de desconto
- $\mathbb{E}[Q(S',A')]$ é o valor esperado considerando todas as ações possíveis no próximo estado

### SARSA Off-Policy X On-Policy

A principal diferença entre as versões Off-Policy e On-Policy do Expected SARSA está nas políticas utilizadas:

- **On-Policy**: Usa a mesma política para agir e aprender.
- **Off-Policy**: Usa duas políticas diferentes:
  - Política comportamental (behavior policy): usada para explorar e coletar experiências
  - Política alvo (target policy): politica que vai ser utilizada quando a behavior já preencheu a Q-table para escolher a ação a ser tomada em cada estado.

Em nossa implementação, testamos três tipos de políticas:

- Epsilon-gulosa: combina exploração aleatória com exploração gulosa.
- Softmax: seleciona ações baseadas em uma distribuição de probabilidade sobre os valores Q.
- Aleatória: seleciona ações uniformemente ao acaso.

## Implementando o SARSA Off-Policy

Nossa implementação do Expected SARSA Off-Policy possui os seguintes componentes principais:

1. **Políticas de aprendizado**:
   - `politica_epsilon_gulosa`: Implementa a política ε-greedy.
   - `politica_softmax`: Implementa a política softmax com temperatura.
   - `politica_aleatoria`: Implementa a política de seleção aleatória.

2. **Algoritmo principal**: 
   - `executar_expected_sarsa`: Implementa o loop principal de treinamento.
   - Parâmetros configuráveis: taxa de aprendizado, desconto, políticas de treinamento e alvo.

3. **Ambientes de teste**:
   - CliffWalking: Ambiente de navegação com precipício.
   - FrozenLake: Ambiente de navegação em lago congelado.
   - Taxi: Ambiente de navegação com coleta e entrega.
   - RaceTrack: Ambiente de corrida customizado.

4. **Métricas**:
   - Recompensa média por episódio.
   - Comparação entre diferentes combinações de políticas.

## Políticas Implementadas

### Epsilon-Gulosa (ε-greedy)

A política epsilon-gulosa equilibra exploração e aproveitamento através de um parâmetro ε:
- Com probabilidade (1-ε): escolhe a ação com maior valor Q (gulosa).
- Com probabilidade ε: escolhe uma ação aleatoriamente.
- Vantagens:
  - Garante exploração contínua;
  - Simples de implementar; e
  - Controle direto do nível de exploração.
- Desvantagens:
  - Exploração uniforme pode ser ineficiente; e
  - ε fixo pode ser subótimo em diferentes fases do aprendizado.

### Softmax (Boltzmann)

A política softmax usa uma distribuição de probabilidade baseada nos valores Q:
- Probabilidade de cada ação é proporcional a exp(Q(s,a)/τ).
- τ é o parâmetro de temperatura que controla a aleatoriedade:
  - τ alto: ações mais equiprováveis (mais exploração).
  - τ baixo: maior probabilidade para ações com valores Q mais altos.
- Vantagens:
  - Exploração mais informada que ε-greedy;
  - Probabilidades proporcionais aos valores Q; e
  - Suave transição entre exploração e aproveitamento.
- Desvantagens:
  - Mais complexa computacionalmente;
  - Sensível à escala dos valores Q; e
  - Ajuste da temperatura pode ser desafiador.

### Aleatória (Random)

A política aleatória seleciona ações com probabilidade uniforme:
- Todas as ações têm igual probabilidade de seleção.
- Usada principalmente como baseline e para exploração pura.
- Vantagens:
  - Máxima exploração do espaço de ações;
  - Útil para coleta inicial de experiências; e
  - Serve como política comportamental off-policy.
- Desvantagens:
  - Nenhum aproveitamento do conhecimento adquirido; e
  - Ineficiente para aprendizado direto.

## Otimização de Hiperparâmetros com Optuna

Para melhorar o desempenho do algoritmo em diferentes ambientes, implementamos um sistema de otimização automática de hiperparâmetros usando a biblioteca Optuna. Esta seção descreve as novas funções adicionadas:

### Funções de Otimização

`objective(trial, env_name, policy_name)`

Função central para busca de hiperparâmetros ótimos usando amostragem.

**Parâmetros:**

- `trial`: Objeto do Optuna para sugestão de valores.
- `env_name`: Nome do ambiente Gymnasium.
- `policy_name`: Política sendo otimizada (epsilon-gulosa/softmax/aleatoria).
- `max_passos`: Quantidade de passos que vai ser utilizado em cada iteração.

### Funcionamento:

1. Sugere valores para taxa de aprendizado (1e-3 a 0.5) e desconto (0.8 a 0.999).
2. Para ε-greedy: sugere ε entre 0.01-0.5.
3. Para softmax: sugere temperatura entre 0.1-10.0.
4. Executa o Expected SARSA com 100 episódios (reduzido para agilizar busca).
5. Retorna a média das últimas 100 recompensas como métrica de qualidade.

`otimizar_hiperparametros(envs, n_trials, max_passos)`

Orquestra o processo de otimização para todos ambientes e políticas.

### Fluxo:

1. Para cada ambiente (FrozenLake, Taxi, etc.):
    - Cria estudo Optuna com sampler TPE (Tree-structured Parzen Estimator).
    - Usa pruner para interromper trials pouco promissores.

2. Para cada política (ε-greedy, softmax, random):
    - Executa n_trials tentativas de otimização.
    - Armazena os melhores parâmetros encontrados.

3. Retorna dicionário hierárquico com melhores parâmetros por ambiente/política.

### Estratégias de Otimização:

- TPESampler: algoritmo de amostragem eficiente para espaços de alta dimensão.

- MedianPruner: interrompe trials com desempenho abaixo da mediana.

- Semente Fixa (42): garante reprodutibilidade dos experimentos.

## Ambientes de Teste

### CliffWalking

Ambiente de grade 4x12 onde o agente deve navegar do ponto inicial até o objetivo evitando cair do penhasco.

![a imagem mostra um GIF do ambiente cliff walking, onde um agente tenta alcançar uma recompensa (representado por um biscoito) sem tentar cair do penhasco, que possui uma posição fixa no ambiente](https://gymnasium.farama.org/_images/cliff_walking.gif)

- **Estados**: 48 posições possíveis.
- **Ações**: 4 (cima, baixo, esquerda, direita).
- **Recompensas**: -1 por passo, -100 por cair do penhasco, 0 ao atingir o objetivo.

### FrozenLake

Ambiente 4x4 onde o agente deve atravessar um lago congelado evitando buracos.

![a imagem mostra um GIF do ambiente frozen lake, onde um agente tenta alcançar uma recompensa (representado por um presente) sem cair em um lago gelado, diferentemente do cliff walking os buracos são posicionados de forma aleatória](https://gymnasium.farama.org/_images/frozen_lake.gif)

- **Estados**: 16 posições possíveis.
- **Ações**: 4 (cima, baixo, esquerda, direita).
- **Recompensas**: 0 por movimento, 1 ao atingir objetivo, episódio termina ao cair em buraco.

### Taxi

Ambiente onde um táxi deve pegar e entregar passageiros em locais específicos.

![a imagem mostra um GIF do ambiente taxi, onde um agente tenta pegar e entregar passageiros em locais específicos, movendo-se em uma grade representando uma cidade](https://gymnasium.farama.org/_images/taxi.gif)

- **Estados**: 500 estados possíveis (25 posições x 5 locais de passageiros x 4 destinos).
- **Ações**: 6 (norte, sul, leste, oeste, pegar, deixar).
- **Recompensas**: -1 por movimento, +20 por entrega bem-sucedida, -10 por ações ilegais.

### RaceTrack

Ambiente customizado simulando uma pista de corrida.

![a imagem mostra um GIF do ambiente RaceTrack, onde um onde o agente deve seguir as pistas enquanto evita colisões com outros veículos](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/racetrack-env.gif)


- **Estados**: Posição (x,y) e velocidade (vx,vy).
- **Ações**: 9 possíveis acelerações (-1,0,1 em cada eixo).
- **Recompensas**: -1 por passo, -5 por colisão, 0 ao cruzar linha de chegada.

## Resultados

## Conclusão

## Agradecimentos