# 🧮 Compiler and RPN (Reverse Polish Notation) Calculator
# Project Phase 4
## Integrantes - Grupo 04
- Gabriel Martins Vicente
- Javier Agustin Aranibar González
- Matheus Paul Lopuch
- Rafael Bonfim Zacco

Este projeto implementa um compilador simples para uma linguagem de expressão personalizada baseada em RPN (Reverse Polish Notation - Notação Polonesa Reversa), incluindo:

- ✅ Análise léxica
- ✅ Análise sintática (Parsing LL(1))
- ✅ Geração de AST (Árvore de Sintaxe Abstrata)
- ✅ Análise semântica e inferência de tipos
- ✅ Interpretação e avaliação
- ✅ Geração de código Assembly para microcontroladores AVR (ATmega328P - Arduino Uno)

---

## ⚙️ Funcionalidades

### 🔹 Análise Léxica (Lexer)
- Responsável por quebrar a entrada da linguagem (código-fonte) em uma sequência de unidades menores chamadas "tokens".
- Identifica e categoriza elementos como números (inteiros e de ponto flutuante), identificadores (nomes de variáveis), operadores (aritméticos, lógicos, comparação), parênteses e palavras-chave reservadas.
- Suporta detecção de números com parte decimal e notação científica (ex: `1.23, 4e-5`).
- Reconhece palavras-chave específicas da linguagem, como `RES, MEM, if, then, else, for, in, to, do.`
- Lida com espaços em branco e comentários de linha (ignorando-os).

### 🔹 Análise Sintática (Parser LL(1))
- Verifica se a sequência de tokens está em conformidade com as regras gramaticais da linguagem, ou seja, se a "sentença" é bem formada.
- Implementa um parser preditivo descendente do tipo LL(1), que significa "Left-to-right, Leftmost derivation, 1 token lookahead".
- Utiliza Conjuntos FIRST e Conjuntos FOLLOW para construir uma Tabela de Parsing LL(1).
- FIRST(X): Contém todos os símbolos terminais que podem iniciar uma derivação de X.
- FOLLOW(X): Contém todos os símbolos terminais que podem aparecer imediatamente após X em alguma forma sentencial.
- A Tabela de Parsing mapeia pares (não-terminal, terminal) para as produções gramaticais apropriadas, guiando o processo de parsing
- Constrói uma Árvore de Sintaxe Abstrata (AST) a partir dos tokens analisados. A AST representa a estrutura hierárquica e o significado da expressão, abstraindo detalhes sintáticos desnecessários.
- É responsável por lidar com a precedência e associatividade de operadores através da própria estrutura da notação RPN e da forma como a AST é construída.

### 🔹 Geração da AST
- A AST é a representação intermediária do código-fonte, crucial para as fases posteriores do compilador.
- Cada nó na AST representa uma construção da linguagem (ex: um número, um identificador, uma operação, uma condicional).
#### Os tipos de nós incluem:
`NumberNode:` Para valores numéricos (inteiros ou flutuantes).

`IdentifierNode:` Para variáveis.

`BinaryOpNode:` Para operações binárias (ex: `+, -, *`).

`IfNode:` Para estruturas condicionais `if-then-else`.

`ForNode:` Para loops `for-in-to-do`.

`StringNode:` Para literais de string.

`MemoryStoreNode:` Para operações de armazenamento na memória (`X MEM`).

`MemoryRetrieveNode:` Para operações de recuperação da memória (`MEM`).

`ResultNode:` Para acessar resultados anteriores (`N RES ou RES`).

- Visualização: Oferece a capacidade de gerar uma representação visual da AST em formato de imagem (`.png`) usando a biblioteca `graphviz`, o que é excelente para depuração e compreensão da estrutura do código.
- Serialização: Permite a exportação da estrutura da AST para formato JSON, facilitando a integração com outras ferramentas ou a análise programática da árvore.

### 🔹 Análise Semântica e Inferência de Tipos
- Esta fase é responsável por verificar o significado e a consistência do programa, indo além da mera correção sintática.
- No seu compilador, a análise semântica está integrada principalmente nos métodos `check_types()` de cada nó da AST e na classe `TypeInference`.
- Verificação de Tipos (check_types() nos nós da AST):
- Cada `ASTNode` (e suas subclasses) possui um método `check_types()` que é invocado recursivamente após a construção da AST.
- Este método verifica se as operações e construções da linguagem estão sendo usadas com os tipos de dados compatíveis.
#### Exemplos de verificações:
Um `BinaryOpNode` (como `+, -, *`) verifica se seus operandos são numéricos (`is_numeric`) e determina o tipo resultante (inteiro se ambos forem inteiros, flutuante caso contrário).

Um `IfNode` verifica se a condição é booleana ou numérica e se os tipos dos ramos `then` e `else` são compatíveis (permitindo casting implícito de `int` para `float`).

Um `ForNode` verifica se os limites do loop (`in e to`) são inteiros.

Um `MemoryStoreNode` garante que apenas valores numéricos podem ser armazenados na memória.

Um `esultNode` (para `N RES`) exige que `N` seja um número inteiro.

- Se uma incompatibilidade de tipo ou uma estrutura semanticamente inválida for detectada, um ValueError é lançado, indicando um "Erro semântico".

- Inferência de Tipos (`TypeInference`):
Esta classe formaliza o processo de dedução dos tipos das expressões.

- Ela utiliza um sistema de regras de inferência de tipos (similar à Lógica de Hoare ou sistemas de tipos formais) para construir "sequentes".
- Um "sequente" é representado como `{contexto} ⊢ expressão : tipo`, onde o `contexto` é um mapeamento de variáveis para seus tipos.
#### As regras de inferência são aplicadas a partir dos nós folha da AST, subindo a árvore:

`var_rule:` Para inferir o tipo de uma variável a partir do contexto.

`num_rule`: Para números literais (`int` ou `float`).

`binop_rule`: Para operações binárias, determinando o tipo resultante com base nos operandos.

`if_rule`: Para expressões condicionais, garantindo a compatibilidade dos tipos dos ramos.

`for_rule`: Para loops `for`, adicionando a variável de iteração ao contexto com o tipo inteiro e inferindo o tipo do corpo.

`mem_store_rule`, `mem_retrieve_rule`, `res_rule`: Para as operações de memória e resultados.

- A sequência de sequentes gerados é impressa, ilustrando o passo a passo da derivação e inferência de tipos.
- O tipo inferido final para a expressão raiz é exibido.
- Este processo garante que o programa não só está sintaticamente correto, mas também faz sentido em termos de tipos de dados.

### 🔹 Interpretação/Avaliação
- Após a análise léxica, sintática e semântica, a AST é percorrida (avaliada) para calcular o resultado da expressão.
- O método `eval_ast(node)` do `Parser` é o interpretador principal.
- Gerencia o estado da memória (`self.memoria`), que armazena um único valor flutuante.
- Mantém um histórico de resultados (`self.resultados`), permitindo o acesso a valores de expressões anteriores através da palavra-chave `RES`.
- Suporta casting implícito entre `int` e `float` para operações numéricas, facilitando a interação entre diferentes tipos numéricos.
- Lida com erros de tempo de execução, como divisão por zero ou acesso a variáveis não declaradas.

### 🔹 Geração de Código Assembly (AVR - ATmega328P)
- Gera `calculadora.asm` com instruções UART.
- Suporte a IEEE 754 (16, 32 ou 64 bits).
- Exibe resultado + representação hexadecimal via serial.

---

## 📥 Como Usar

### 🔧 Pré-requisitos

Instale as bibliotecas necessárias:
```bash
pip install graphviz tabulate colorama
```
Instale o Graphviz no sistema:

[Site oficial](https://graphviz.org/download/)

Após instalar, descompacte o arquivo e vá até a pasta bin. Copie o local e adicione ao PATH.

---

### ▶️ Executando o Compilador

Execute algum dos arquivos de expressões, por exemplo o teste1.txt
```bash
python compilador.py teste1.txt
```

### 📋 O que será exibido
- Tokens gerados

- Conjuntos FIRST e FOLLOW

- Tabela de Parsing LL(1)

- Derivação de tipos

- AST em texto indentado

- AST em .png (visual)

- AST em JSON

- Resultado + IEEE 754 (hex)

- Código calculadora.asm gerado

### 🔧 Compilando o Código Assembly

Para mostrar no Serial Monitor do Arduino, você deve utilizar o arquivo .bat que está disponibilizado 

[Compilador automático](./compilar.bat)

Antes de executar, verifique a porta COM que está conectado o seu Arduino. (Por padrão no .bat está a COM3).

---

## 🗂️ Estrutura do Código

### 🔸 `Token`
- `namedtuple` para armazenar informações dos tokens:
  - `valor`, `tipo`, `linha`, `coluna`.

### 🔸 `FirstSet / FollowSet`
- Implementação dos conjuntos FIRST e FOLLOW para construção da **Tabela de Parsing LL(1)**.

### 🔸 `ParsingTable`
- Mapeia produções gramaticais com base nos conjuntos **FIRST** e **FOLLOW**.

### 🔸 `Lexer`
- Converte a entrada do código-fonte em uma **lista de tokens** reconhecíveis.

### 🔸 `ASTNode` e suas Subclasses
- Representam a **Árvore de Sintaxe Abstrata**.
- Exemplo de subclasses: `NumberNode`, `BinaryOpNode`, `IfNode`, `ForNode`, `MemoryNode`, etc.

### 🔸 `Parser`
- `parse()` e `parse_paren()` constroem a **AST** com base nos tokens.
- `eval_ast()` percorre e **interpreta** a AST.
- `eval_op()` executa **operações aritméticas e lógicas**.

### 🔸 `TypeInference`
- Executa regras formais de **inferência de tipos**.
- Exibe os **sequentes** gerados durante o processo.

### 🔸 Assembly Generator
- Função `escrever_serial()` gera o código de saída em **Assembly UART**, formatando:
  - A operação,
  - O resultado,
  - O tipo de dado,
  - E sua representação **IEEE 754 hexadecimal**.

