"""

Phase 4 Project - Group 04

Students: 
- Gabriel Martins Vicente
- Javier Agustin Aranibar González  
- Matheus Paul Lopuch
- Rafael Bonfim Zacco

"""

import sys
import re
import math
import platform
import struct
from collections import namedtuple
from graphviz import Digraph
import json
from tabulate import tabulate
from colorama import init, Fore, Back, Style
from typing import Dict, List, Set, Tuple, Optional, Union

# Inicializa o colorama
init()

# Detecta arquitetura para precisão
bits = platform.architecture()[0]
if bits == '16bit':
    PRECISAO = 'half'  # 16 bits
elif bits == '32bit':
    PRECISAO = 'single'  # 32 bits
elif bits == '64bit':
    PRECISAO = 'double'  # 64 bits
else:
    PRECISAO = 'quadruple'  # 128 bits (simulado)

# Operadores e palavras-chave
OPERATORS = {'+', '-', '*', '/', '%', '^', '|', '<', '>', '==', '!=', '<=', '>='}
KEYWORDS = {'RES', 'MEM', 'if', 'then', 'else', 'for', 'in', 'to', 'do'}

Token = namedtuple('Token', ['value', 'type', 'line', 'col'])

# Funções de conversão IEEE 754
def float_to_half_ieee754(f):
    """
    Converte um número float para formato IEEE 754 half-precision (16 bits)
    """
    if f == 0.0: return 0x0000
    if f < 0.0 and f == 0.0: return 0x8000
    if f == float('inf'): return 0x7C00
    if f == float('-inf'): return 0xFC00
    if math.isnan(f): return 0x7E00
    
    sinal = 0x8000 if f < 0 else 0
    f = abs(f)
    
    if f >= 2**(-14):
        expoente = math.floor(math.log2(f))
        mantissa = f / (2**expoente) - 1.0
        expoente_ajustado = expoente + 15
        
        if expoente_ajustado < 1: 
            return f"0x{sinal:04x}"
            if expoente_ajustado > 30: 
                return f"0x{(sinal | 0x7C00):04x}"
        
        bits_mantissa = int(mantissa * 1024 + 0.5)
        half = sinal | ((expoente_ajustado & 0x1F) << 10) | (bits_mantissa & 0x3FF)
    else:
        mantissa = f / (2**(-14))
        bits_mantissa = int(mantissa * 1024 + 0.5)
        half = sinal | (bits_mantissa & 0x3FF)
        
    return f"0x{half:04x}"

def half_ieee754_to_float(h):
    """
    Converte um número IEEE 754 half-precision (16 bits) para float
    """
    sinal = -1.0 if (h & 0x8000) else 1.0
    expoente = (h >> 10) & 0x1F
    mantissa = h & 0x3FF
    
    if expoente == 0:
        if mantissa == 0: return 0.0 * sinal
        return sinal * (mantissa / 1024.0) * (2 ** -14)
    elif expoente == 31:
        if mantissa == 0: return float('inf') * sinal
        return float('nan')
    
    return sinal * (1.0 + mantissa / 1024.0) * (2 ** (expoente - 15))

def add_half_precision(a, b):
    """
    Soma dois números em formato IEEE 754 half-precision
    """
    fa, fb = half_ieee754_to_float(a), half_ieee754_to_float(b)
    return float_to_half_ieee754(fa + fb)

def sub_half_precision(a, b):
    """
    Subtração de dois números em formato IEEE 754 half-precision
    """
    fa, fb = half_ieee754_to_float(a), half_ieee754_to_float(b)
    return float_to_half_ieee754(fa - fb)

def mul_half_precision(a, b):
    """
    Multiplicação de dois números em formato IEEE 754 half-precision
    """
    fa, fb = half_ieee754_to_float(a), half_ieee754_to_float(b)
    return float_to_half_ieee754(fa * fb)

def div_half_precision(a, b):
    """
    Divisão de dois números em formato IEEE 754 half-precision
    """
    fa, fb = half_ieee754_to_float(a), half_ieee754_to_float(b)
    if fb == 0: return 0x7C00 if fa >= 0 else 0xFC00
    return float_to_half_ieee754(fa / fb)

def power_half_precision(a, b):
    """
    Potenciação em formato IEEE 754 half-precision
    """
    fa, fb = half_ieee754_to_float(a), half_ieee754_to_float(b)
    try:
        # Se o expoente for 0 e a base não for 0, retorna 1.0
        if fb == 0.0 and fa != 0.0:
            return float_to_half_ieee754(1.0)
        return float_to_half_ieee754(fa ** fb)
    except:
        return 0x7E00  # NaN para casos de erro

def mod_half_precision(a, b):
    """
    Operação de módulo em formato IEEE 754 half-precision
    """
    fa, fb = half_ieee754_to_float(a), half_ieee754_to_float(b)
    if fb == 0: return 0x7E00
    return float_to_half_ieee754(fa % fb)

class FirstSet:
    def __init__(self):
        self.sets = {
            'S': {'(', 'número', 'identificador', 'MEM', 'RES', 'if', 'for'},
            'EXPR': {'(', 'número', 'identificador', 'MEM', 'RES', 'if', 'for'},
            'OPERAND': {'número', 'identificador', 'MEM', 'RES'},
            'OPERATOR': set(OPERATORS),
            'NUM': {'número'}
        }
    
    def get(self, non_terminal):
        return self.sets.get(non_terminal, set())

class FollowSet:
    def __init__(self):
        self.sets = {
            'S': {'$'},
            'EXPR': {')', 'then', 'else', 'do', '$', 'MEM', 'RES'},
            'OPERAND': {'(', 'número', 'identificador', 'MEM', '+', '-', '*', '/', '%', '^', '|', '<', '>', '==', '!=', '<=', '>='},
            'OPERATOR': {')'},
            'NUM': {')', 'MEM', 'RES', '+', '-', '*', '/', '%', '^', '|', '<', '>', '==', '!=', '<=', '>='}
        }
    
    def get(self, non_terminal):
        return self.sets.get(non_terminal, set())

class ParsingTable:
    def __init__(self):
        self.first_sets = FirstSet()
        self.follow_sets = FollowSet()
        self.table = {}
        self.initialize_table()
    
    def initialize_table(self):
        non_terminals = ['S', 'EXPR', 'OPERAND', 'OPERATOR', 'NUM']
        terminals = set()
        
        for first_set in self.first_sets.sets.values():
            terminals.update(first_set)
        for follow_set in self.follow_sets.sets.values():
            terminals.update(follow_set)
        
        terminals.update(OPERATORS)
        terminals.add('(')
        terminals.add(')')
        terminals.add('$')
        terminals.add('número')
        terminals.add('identificador')
        terminals.add('palavra-chave')

        for nt in non_terminals:
            self.table[nt] = {}
            for t in terminals:
                self.table[nt][t] = None
        
        self.fill_table()
    
    def fill_table(self):
        def map_terminal(token_type_or_value):
            if token_type_or_value == 'INT' or token_type_or_value == 'REAL':
                return 'número'
            elif token_type_or_value == 'ID':
                return 'identificador'
            elif token_type_or_value in KEYWORDS:
                return token_type_or_value
            elif token_type_or_value in OPERATORS:
                return 'operador'
            return token_type_or_value

        for t in self.first_sets.get('EXPR'):
            self.table['S'][map_terminal(t)] = ['EXPR']
        
        # RPN production: ( EXPR EXPR OPERATOR )
        self.table['EXPR']['('] = ['(', 'EXPR', 'EXPR', 'OPERATOR', ')']

        self.table['EXPR']['if'] = ['(', 'if', 'EXPR', 'then', 'EXPR', 'else', 'EXPR', ')']
        
        self.table['EXPR']['for'] = ['(', 'for', 'ID', 'in', 'EXPR', 'to', 'EXPR', 'do', 'EXPR', ')']
        
        for t in self.first_sets.get('OPERAND'):
            self.table['EXPR'][map_terminal(t)] = ['OPERAND']
        
        for t in self.first_sets.get('NUM'):
            self.table['OPERAND'][map_terminal(t)] = ['NUM']
        
        self.table['OPERAND']['identificador'] = ['ID']
        
        self.table['OPERAND']['MEM'] = ['MEM']
        
        self.table['OPERAND']['RES'] = ['RES']
        
        # This part should be correct as OPERATOR is a terminal itself.
        for op_val in OPERATORS:
            self.table['OPERATOR'][op_val] = [op_val] # Changed 'operador' to op_val directly
        
        self.table['NUM']['número'] = ['INT']
    
    def print_table(self):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Tabela de Parsing LL(1):{Style.RESET_ALL}")
        
        terminals = set()
        for nt in self.table:
            terminals.update(self.table[nt].keys())
        terminals = sorted(list(terminals))
        
        headers = [f"{Fore.GREEN}NT/T{Style.RESET_ALL}"] + [f"{Fore.YELLOW}{t}{Style.RESET_ALL}" for t in terminals]
        table_data = []
        
        for nt in sorted(self.table.keys()):
            row = [f"{Fore.GREEN}{nt}{Style.RESET_ALL}"]
            for t in terminals:
                prod = self.table[nt].get(t)
                if prod is None:
                    row.append('')
                else:
                    row.append(' '.join(prod))
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()

def float_to_ieee754(f):
    f = float(f) 
    if PRECISAO == 'half':
        # Tratamento de casos especiais
        if f == 0.0: return 0x0000  # Zero positivo
        if f < 0.0 and f == 0.0: return 0x8000  # Zero negativo
        if f == float('inf'): return 0x7C00  # Infinito positivo
        if f == float('-inf'): return 0xFC00  # Infinito negativo
        if math.isnan(f): return 0x7E00  # NaN (Not a Number)
        
        # Extrair sinal, expoente e mantissa
        sinal = 0x8000 if f < 0 else 0
        f = abs(f)
        
        # Normalizar para o formato IEEE 754
        if f >= 2.0 ** (-14):  # Valores normalizados
            expoente = math.floor(math.log2(f))
            mantissa = f / (2 ** expoente) - 1.0
            expoente_ajustado = expoente + 15  # Bias de 15
            
            # Verificar limites
            if expoente_ajustado < 0: return sinal  # Underflow
            if expoente_ajustado > 31: return sinal | 0x7C00  # Overflow
            
            # Calcular os bits da mantissa (10 bits)
            bits_mantissa = int(mantissa * 1024 + 0.5)
            half = sinal | ((expoente_ajustado & 0x1F) << 10) | (bits_mantissa & 0x3FF)
        else:  # Valores desnormalizados
            mantissa = f / (2 ** (-14))
            bits_mantissa = int(mantissa * 1024 + 0.5)
            half = sinal | (bits_mantissa & 0x3FF)
        
        return f"0x{half:04x}" # Retorna como string hexadecimal
    elif PRECISAO == 'single':
        return '0x' + struct.pack('>f', f).hex()
    elif PRECISAO == 'double':
        return '0x' + struct.pack('>d', f).hex()
    else:
        return hex(int(f)) 

def lexer(line, line_num=1):
    tokens = []
    i = 0
    estado = 'INICIO'
    while i < len(line):
        c = line[i]
        if estado == 'INICIO':
            if c.isspace():
                i += 1
                continue
            elif c == '(' or c == ')':
                tokens.append(Token(c, 'parêntese', line_num, i))
                i += 1
            elif c == '"':
                start = i
                estado = 'STRING'
                i += 1
            elif any(line.startswith(op, i) for op in sorted(OPERATORS, key=len, reverse=True)):
                for op in sorted(OPERATORS, key=len, reverse=True):
                    if line.startswith(op, i):
                        tokens.append(Token(op, 'operador', line_num, i))
                        i += len(op)
                        break
            elif c.isdigit() or (c == '.' and i+1 < len(line) and line[i+1].isdigit()):
                start = i
                estado = 'NUMERO'
            elif c.isalpha():
                start = i
                estado = 'ID'
            else:
                tokens.append(Token(c, 'desconhecido', line_num, i))
                i += 1
        elif estado == 'STRING':
            if c == '"':
                tokens.append(Token(line[start+1:i], 'string', line_num, start))
                i += 1
                estado = 'INICIO'
            else:
                i += 1
        elif estado == 'NUMERO':
            j = i
            while j < len(line) and (line[j].isdigit() or line[j] == '.' or \
                                     (line[j] in 'eE' and j+1 < len(line) and \
                                      (line[j+1].isdigit() or line[j+1] in '+-'))):
                j += 1
            tokens.append(Token(line[start:j], 'número', line_num, start))
            i = j
            estado = 'INICIO'
        elif estado == 'ID':
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] == '_'):
                j += 1
            lex = line[start:j]
            if lex in KEYWORDS:
                classe = 'palavra-chave'
            else:
                classe = 'identificador'
            tokens.append(Token(lex, classe, line_num, start))
            i = j
            estado = 'INICIO'
    return tokens

class Parser:
    def __init__(self, tokens, memoria, resultados):
        self.tokens = tokens
        self.pos = 0
        self.memoria = memoria
        self.resultados = resultados
        self.vars = {}
        self.first_sets = FirstSet()
        self.follow_sets = FollowSet()
        self.allow_casting = True

    def set_allow_casting(self, allow):
        self.allow_casting = allow

    def at(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek_next(self):
        return self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None

    def eat(self, expected_type=None, expected_value=None):
        t = self.at()
        if t is None:
            raise ValueError('Fim inesperado da entrada')
        if expected_type and t.type != expected_type:
            raise ValueError(f'Esperado tipo "{expected_type}", encontrou "{t.type}" (valor: "{t.value}") na linha {t.line}, coluna {t.col}')
        if expected_value and t.value != expected_value:
            raise ValueError(f'Esperado valor "{expected_value}", encontrou "{t.value}" (tipo: "{t.type}") na linha {t.line}, coluna {t.col}')
        self.pos += 1
        return t

    def check_first(self, non_terminal, token):
        first_set = self.first_sets.get(non_terminal)
        if token is None:
            return False
        if token.type == 'número':
            return 'número' in first_set
        elif token.type == 'identificador':
            return 'identificador' in first_set
        elif token.type == 'palavra-chave':
            return token.value in first_set
        elif token.type == 'operador':
            return token.value in first_set
        return token.value in first_set

    def check_follow(self, non_terminal, token):
        follow_set = self.follow_sets.get(non_terminal)
        if token is None:
            return '$' in follow_set
        if token.type == 'número':
            return 'número' in follow_set
        elif token.type == 'identificador':
            return 'identificador' in follow_set
        elif token.type == 'palavra-chave':
            return token.value in follow_set
        elif token.type == 'operador':
            return token.value in follow_set
        return token.value in follow_set

    def eval_ast(self, node):
        try:
            if isinstance(node, NumberNode):
                return node.value
            elif isinstance(node, IdentifierNode):
                if node.value not in self.vars:
                    raise ValueError(f"Variável '{node.value}' não declarada")
                return self.vars[node.value]
            elif isinstance(node, BinaryOpNode):
                a = self.eval_ast(node.children[0])
                b = self.eval_ast(node.children[1])
                
                if self.allow_casting:
                    if isinstance(a, int) and isinstance(b, float):
                        a = float(a)
                    elif isinstance(a, float) and isinstance(b, int):
                        b = float(b)
                result = self.eval_op(a, b, node.value)
                # Only append if it's not a boolean comparison, or if we want to store 0.0/1.0
                if node.value not in ['>', '<', '==', '!=', '<=', '>=']:
                    self.resultados.append(result)
                return result
            elif isinstance(node, IfNode):
                cond = self.eval_ast(node.children[0])
                if not (isinstance(cond, bool) or isinstance(cond, (int, float))):
                    raise ValueError(f"Condição do 'if' deve ser booleana ou numérica, mas encontrou tipo {type(cond).__name__}")
                
                # Evaluate the appropriate branch and append its result if it's numeric
                if bool(cond):
                    result = self.eval_ast(node.children[1])
                else:
                    result = self.eval_ast(node.children[2])
                
                # Check if the result is numeric before appending to `resultados`
                if isinstance(result, (int, float)):
                    self.resultados.append(result)
                return result

            elif isinstance(node, ForNode):
                var = node.children[0].value
                ini = int(self.eval_ast(node.children[1]))
                fim = int(self.eval_ast(node.children[2]))
                res = 0.0
                old_value = self.vars.get(var)
                try:
                    for v in range(ini, fim + 1):
                        self.vars[var] = v
                        res = self.eval_ast(node.children[3])
                    # Only append if the loop body produced a numeric result
                    if isinstance(res, (int, float)):
                        self.resultados.append(res)
                    return res
                finally:
                    if old_value is not None:
                        self.vars[var] = old_value
                    else:
                        del self.vars[var]
            elif isinstance(node, MemoryStoreNode):
                value_to_store = self.eval_ast(node.children[0])
                self.memoria = float(value_to_store)
                self.resultados.append(self.memoria) # Always append memory store results
                return self.memoria
            elif isinstance(node, MemoryRetrieveNode):
                result = self.memoria
                self.resultados.append(result) # Always append memory retrieve results
                return result
            elif isinstance(node, ResultNode):
                n_node = node.children[0]
                n = int(self.eval_ast(n_node))
                
                if n < 0 or n >= len(self.resultados):
                    raise ValueError(f"Índice RES({n}) fora dos limites. Resultados disponíveis: {len(self.resultados)}")
                
                # The index is negative because RES(N) means the Nth result *from the last*.
                # If RES(0) is the last, RES(1) is the second to last etc.
                result = self.resultados[len(self.resultados) - 1 - n] 
                self.resultados.append(result)
                return result
            elif isinstance(node, StringNode):
                return node.value
            else:
                raise ValueError(f'Nó AST desconhecido: {type(node)}')
        except Exception as e:
            print(f"{Fore.RED}Erro em tempo de execução: {str(e)}{Style.RESET_ALL}")
            raise

    def eval_op(self, a, b, op):
        try:
            if op in ['+', '-', '*', '/', '|', '%', '^']:
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    raise ValueError(f"Operador '{op}' requer operandos numéricos")
            
            if op == '+': return a + b
            if op == '-': return a - b
            if op == '*': return a * b
            if op == '|': return a / b
            if op == '/': 
                if b == 0:
                    raise ValueError("Divisão por zero")
                return float(int(a) // int(b)) 
            if op == '%': 
                if b == 0:
                    raise ValueError("Módulo por zero")
                return float(int(a) % int(b))
            if op == '^': 
                if not isinstance(b, int):
                     raise ValueError("Expoente para '^' deve ser inteiro")
                if a == 0 and b == 0:
                    return 1.0
                if b == 0:
                    return 1.0
                if a == 0:
                    return 0.0
                if a < 0 and b < 0:
                    return (-(-a) ** b)
                if a < 0:
                    return (-(-a) ** b)
                if b < 0:
                    return 1.0 / (-a ** -b)
                if a == 1:
                    return 1.0
                if b == 1:
                    return a
                if a == -1 and b % 2 == 0:
                    return 1.0
                if a == -1 and b % 2 != 0:
                    return -1.0
                if b % 2 == 0:
                    return (a ** b)
                else:
                    return (a ** b)
            if op in ['>', '<', '==', '!=', '<=', '>=']:
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    raise ValueError(f"Operador '{op}' requer operandos numéricos")
                if op == '>': return float(a > b)
                if op == '<': return float(a < b)
                if op == '==': return float(a == b)
                if op == '!=': return float(a != b)
                if op == '<=': return float(a <= b)
                if op == '>=': return float(a >= b)
            raise ValueError(f'Operador desconhecido: {op}')
        except Exception as e:
            print(f"{Fore.RED}Erro na operação '{op}': {str(e)}{Style.RESET_ALL}")
            raise

    def parse(self):
        t = self.at()
        if t is None:
            raise ValueError('Expressão vazia')
        
        if t.value == '(':
            return self.parse_paren()
        elif t.type == 'número':
            token = self.eat('número')
            try:
                value = float(token.value)
                # Preserva o tipo float se o número foi escrito com ponto decimal
                if '.' in token.value:
                    return NumberNode(value)
                # Converte para int apenas se não tiver ponto decimal
                if value.is_integer():
                    value = int(value)
                return NumberNode(value)
            except ValueError:
                raise ValueError(f"Valor numérico inválido: '{token.value}' na linha {token.line}, coluna {token.col}")
        elif t.type == 'string':
            token = self.eat('string')
            return StringNode(token.value)
        elif t.type == 'palavra-chave' and t.value == 'MEM':
            self.eat('palavra-chave', 'MEM')
            return MemoryRetrieveNode()
        elif t.type == 'palavra-chave' and t.value == 'RES':
            # This path is for a standalone 'RES' without parentheses, e.g., just 'RES' on a line
            # If (N RES) and (RES) are the only forms, this can be removed.
            # Assuming (RES) should default to 0 for N if it's hit this way.
            self.eat('palavra-chave', 'RES')
            return ResultNode(NumberNode(0)) 
        elif t.type == 'identificador':
            token = self.eat('identificador')
            node = IdentifierNode(token.value)
            return node
        else:
            raise ValueError(f'Token inesperado: {t} na linha {t.line}, coluna {t.col}.')

    def parse_paren(self):
        self.eat('parêntese', '(')
        t = self.at()
        
        if t is None:
            raise ValueError("Expressão incompleta após '('")

        if t.type == 'palavra-chave' and t.value == 'if':
            self.eat('palavra-chave', 'if')
            if_node = IfNode()
            if_node.add_child(self.parse())
            self.eat('palavra-chave', 'then')
            if_node.add_child(self.parse())
            self.eat('palavra-chave', 'else')
            if_node.add_child(self.parse())
            self.eat('parêntese', ')')
            return if_node
            
        elif t.type == 'palavra-chave' and t.value == 'for':
            self.eat('palavra-chave', 'for')
            for_node = ForNode()
            var_token = self.eat('identificador')
            var_name = var_token.value
            for_node.add_child(IdentifierNode(var_name))
            self.eat('palavra-chave', 'in')
            for_node.add_child(self.parse())
            self.eat('palavra-chave', 'to')
            for_node.add_child(self.parse())
            self.eat('palavra-chave', 'do')
            
            old_var_value = self.vars.get(var_name, None)
            self.vars[var_name] = 0
            
            body_node = self.parse()
            for_node.add_child(body_node)
            
            if old_var_value is not None:
                self.vars[var_name] = old_var_value
            else:
                del self.vars[var_name]
            
            self.eat('parêntese', ')')
            return for_node

        elif t.type == 'palavra-chave' and t.value == 'MEM':
            # This is for (MEM) meaning retrieve, not store
            if self.peek_next() and self.peek_next().value == ')':
                self.eat('palavra-chave', 'MEM')
                self.eat('parêntese', ')')
                return MemoryRetrieveNode()

        elif t.type == 'palavra-chave' and t.value == 'RES':
            # This is for (RES) meaning last result (N=0)
            if self.peek_next() and self.peek_next().value == ')':
                self.eat('palavra-chave', 'RES')
                self.eat('parêntese', ')')
                return ResultNode(NumberNode(0)) 

        # --- RPN Parsing Logic ---
        # For RPN, we expect operands first, then the operator.

        # Parse the first operand
        first_operand_node = self.parse()

        # Check if the next token is MEM (for store) or RES (for retrieve with N)
        current_token = self.at()
        if current_token is None:
            raise ValueError("Expressão incompleta. Esperado mais tokens ou ')'")

        if current_token.type == 'palavra-chave' and current_token.value == 'MEM':
            self.eat('palavra-chave', 'MEM')
            self.eat('parêntese', ')')
            mem_store_node = MemoryStoreNode()
            mem_store_node.add_child(first_operand_node)
            return mem_store_node
        
        elif current_token.type == 'palavra-chave' and current_token.value == 'RES':
            self.eat('palavra-chave', 'RES')
            self.eat('parêntese', ')')
            if not isinstance(first_operand_node, NumberNode):
                raise ValueError(f"Esperado um número para N em (N RES), mas encontrou {type(first_operand_node).__name__} na linha {first_operand_node.line}, coluna {first_operand_node.col}")
            res_node = ResultNode(first_operand_node) 
            return res_node

        # If not MEM or RES, assume it's a binary operation in RPN form.
        # Parse the second operand
        second_operand_node = self.parse()

        # Now expect the operator
        op_token = self.eat('operador')
        
        if op_token is None or not self.check_first('OPERATOR', op_token):
            raise ValueError(f'Operador inesperado: {op_token}. Esperado um dos operadores em FIRST(OPERATOR): {self.first_sets.get("OPERATOR")}')
        
        bin_op = BinaryOpNode(op_token.value)
        bin_op.add_child(first_operand_node)
        bin_op.add_child(second_operand_node)
        self.eat('parêntese', ')')
        return bin_op

def escrever_serial(asm, operacao, resultado, ieee_hex, tipo_str):
    """
    Escreve o código assembly para enviar a expressão e o resultado via UART
    no formato: (expressão) = resultado [IEEE754: hex]
    """
    resultado_float = float(resultado) 
    
    # Formatar o resultado para string (com casas decimais limitadas para clareza)
    # Ex: 4.0 -> "4", 3.14159 -> "3.14"
    if isinstance(resultado_float, float) and resultado_float.is_integer():
        resultado_str = str(int(resultado_float))
    else:
        resultado_str = f"{resultado_float:.2f}".rstrip('0').rstrip('.') # Limita a 2 casas, remove .0

    # Escrever comentário no .asm
    asm.write(f"\n    ; Expressão: {operacao} = {resultado_str} [IEEE754: {ieee_hex}]\n")
    
    # Enviar a expressão original
    for char in operacao:
        asm.write(f"""
    LDI R16, 0x{ord(char):02X}
    RCALL uart_envia_byte
""")

    # Enviar o sinal de igual e espaço
    asm.write("""
    LDI R16, 0x20  ; ' '
    RCALL uart_envia_byte
    LDI R16, 0x3D  ; '='
    RCALL uart_envia_byte
    LDI R16, 0x20  ; ' '
    RCALL uart_envia_byte
""")
    
    # Enviar o resultado formatado
    for char in resultado_str:
        asm.write(f"""
    LDI R16, 0x{ord(char):02X}
    RCALL uart_envia_byte
""")

    # Enviar o formato IEEE 754: " [IEEE754: "
    asm.write("""
    LDI R16, 0x20  ; ' '
    RCALL uart_envia_byte
    LDI R16, 0x5B  ; '['
    RCALL uart_envia_byte
    LDI R16, 0x49  ; 'I'
    RCALL uart_envia_byte
    LDI R16, 0x45  ; 'E'
    RCALL uart_envia_byte
    LDI R16, 0x45  ; 'E'
    RCALL uart_envia_byte
    LDI R16, 0x45  ; 'E'
    RCALL uart_envia_byte
    LDI R16, 0x37  ; '7'
    RCALL uart_envia_byte
    LDI R16, 0x35  ; '5'
    RCALL uart_envia_byte
    LDI R16, 0x34  ; '4'
    RCALL uart_envia_byte
    LDI R16, 0x3A  ; ':'
    RCALL uart_envia_byte
    LDI R16, 0x20  ; ' '
    RCALL uart_envia_byte
""")
    
    # Enviar o valor IEEE 754 em hex
    for char in ieee_hex:
        asm.write(f"""
    LDI R16, 0x{ord(char):02X}
    RCALL uart_envia_byte
""")
    
    # Enviar o fechamento dos colchetes
    asm.write("""
    LDI R16, 0x5D  ; ']'
    RCALL uart_envia_byte
""")
    
    # Enviar nova linha (CR e LF)
    asm.write("""
    LDI R16, 13  ; CR
    RCALL uart_envia_byte
    LDI R16, 10  ; LF
    RCALL uart_envia_byte
    
    ; Delay para visualização da linha completa
    RCALL delay_ms
""")

class Type:
    def __init__(self, name, is_numeric=False):
        self.name = name
        self.is_numeric = is_numeric

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Type) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def can_cast_to(self, other):
        if self == other:
            return True
        if self.name == 'int' and other.name == 'float':
            return True
        return False

INT_TYPE = Type('int', True)
FLOAT_TYPE = Type('float', True)
STRING_TYPE = Type('string', False)
BOOL_TYPE = Type('bool', False)
VOID_TYPE = Type('void', False)

class ASTNode:
    def __init__(self, value=None):
        self.value = value
        self.children = []
        self.type = None
        self.line = None
        self.col = None

    def add_child(self, child):
        self.children.append(child)

    def check_types(self):
        return self.type

    def visualize(self, dot=None, parent=None):
        if dot is None:
            dot = Digraph(comment='AST')
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='filled', fontname='Arial')
            dot.attr('edge', fontname='Arial')
        
        node_id = str(id(self))
        
        color_map = {
            NumberNode: '#FFB6C1', IdentifierNode: '#98FB98', BinaryOpNode: '#87CEEB',
            IfNode: '#DDA0DD', ForNode: '#F0E68C', StringNode: '#FFA07A',
            MemoryStoreNode: '#ADD8E6', MemoryRetrieveNode: '#87CEFA', ResultNode: '#FFE4B5',
        }
        color = color_map.get(type(self), '#E6E6FA')
        
        text_color_map = {
            INT_TYPE: '#0000FF', FLOAT_TYPE: '#008000', BOOL_TYPE: '#800080',
            STRING_TYPE: '#FF4500', VOID_TYPE: '#A9A9A9',
        }
        text_color = text_color_map.get(self.type, '#000000')
        
        label_value = str(self.value) if self.value is not None else self.__class__.__name__
        label = f"{label_value}\\nType: {self.type if self.type else '?'}"
        
        dot.node(node_id, label, fillcolor=color, fontcolor=text_color)
        
        if parent is not None:
            dot.edge(parent, node_id)
        
        for child in self.children:
            child.visualize(dot, node_id)
        
        return dot

class NumberNode(ASTNode):
    def __init__(self, value):
        super().__init__(value)
        if isinstance(value, int):
            self.type = INT_TYPE
        elif isinstance(value, float):
            self.type = FLOAT_TYPE
        else:
            raise ValueError(f"Invalid type for number: {type(value)}")

    def check_types(self):
        return self.type

class IdentifierNode(ASTNode):
    def __init__(self, name):
        super().__init__(name)
        self.type = None

    def check_types(self, parser=None):
        if self.type is None:
            self.type = INT_TYPE
        if not isinstance(self.type, Type):
            raise ValueError(f"Invalid type object for identifier '{self.value}': {self.type}")
        # Verifica se a variável está declarada durante a análise de tipos
        if parser and self.value not in parser.vars:
            raise ValueError(f"Variável '{self.value}' não declarada")
        return self.type

class BinaryOpNode(ASTNode):
    def __init__(self, op):
        super().__init__(op)
        self.type = None

    def check_types(self):
        if len(self.children) != 2:
            raise ValueError(f"Operator '{self.value}' requires exactly 2 operands")
        
        left_type = self.children[0].check_types()
        right_type = self.children[1].check_types()
        
        if not left_type.is_numeric or not right_type.is_numeric:
            raise ValueError(f"Operands for '{self.value}' must be numeric, but got {left_type} and {right_type}")
        
        if self.value in ['/', '|']:
            self.type = FLOAT_TYPE
        elif self.value in ['>', '<', '==', '!=', '<=', '>=']:
            self.type = BOOL_TYPE
        elif self.value == '^':
            if right_type != INT_TYPE:
                raise ValueError("Exponent for '^' must be an integer")
            if left_type == INT_TYPE and right_type == INT_TYPE:
                self.type = INT_TYPE
            else:
                self.type = FLOAT_TYPE
        elif left_type == FLOAT_TYPE or right_type == FLOAT_TYPE:
            self.type = FLOAT_TYPE
        else:
            self.type = INT_TYPE
        return self.type

class IfNode(ASTNode):
    def __init__(self):
        super().__init__('if')
        self.type = None

    def check_types(self):
        if len(self.children) != 3:
            raise ValueError("If statement requires condition, 'then' branch, and 'else' branch")
        
        cond_type = self.children[0].check_types()
        if not (cond_type == BOOL_TYPE or cond_type.is_numeric):
            raise ValueError(f"Condition of 'if' must be boolean or numeric, but got {cond_type}")
        
        then_type = self.children[1].check_types()
        else_type = self.children[2].check_types()
        
        if then_type.can_cast_to(else_type):
            self.type = else_type
        elif else_type.can_cast_to(then_type):
            self.type = then_type
        else:
            raise ValueError(f"Incompatible types in 'then' ({then_type}) and 'else' ({else_type}) branches of 'if' statement")
        
        return self.type

class ForNode(ASTNode):
    def __init__(self):
        super().__init__('for')
        self.type = None

    def check_types(self):
        if len(self.children) != 4:
            raise ValueError("For loop requires variable, start, end, and body expressions")
        
        var_node = self.children[0]
        var_node.type = INT_TYPE 

        start_type = self.children[1].check_types()
        end_type = self.children[2].check_types()
        
        if not (start_type == INT_TYPE and end_type == INT_TYPE):
            raise ValueError("Start and end expressions of 'for' loop must be integers")
        
        body_type = self.children[3].check_types()
        self.type = body_type
        return self.type

class StringNode(ASTNode):
    def __init__(self, value):
        super().__init__(value)
        self.type = STRING_TYPE

    def check_types(self):
        return self.type

class MemoryStoreNode(ASTNode):
    def __init__(self):
        super().__init__('MEM_STORE')
        self.type = VOID_TYPE

    def check_types(self):
        if len(self.children) != 1:
            raise ValueError("Memory store node requires exactly one value to store")
        
        value_type = self.children[0].check_types()
        if not value_type.is_numeric:
            raise ValueError(f"Only numeric values can be stored in MEM, but got {value_type}")
        
        self.type = FLOAT_TYPE
        return self.type

class MemoryRetrieveNode(ASTNode):
    def __init__(self):
        super().__init__('MEM_RETRIEVE')
        self.type = FLOAT_TYPE

    def check_types(self):
        if len(self.children) != 0:
            raise ValueError("Memory retrieve node takes no arguments")
        self.type = FLOAT_TYPE
        return self.type

class ResultNode(ASTNode):
    def __init__(self, offset_node: NumberNode):
        super().__init__('RES')
        if not isinstance(offset_node, NumberNode):
            raise ValueError("O offset para RES deve ser um número")
        self.add_child(offset_node)
        self.type = None

    def check_types(self):
        if len(self.children) != 1:
            raise ValueError("Result node requires exactly one offset (N)")
        
        offset_type = self.children[0].check_types()
        if offset_type != INT_TYPE:
            raise ValueError("Offset for RES must be an integer")
        self.type = FLOAT_TYPE  # RES sempre retorna float
        return self.type

    def eval_ast(self, parser):
        n = int(self.children[0].value)
        if n < 0:
            raise ValueError(f"Índice RES({n}) deve ser não negativo")
        if n >= len(parser.resultados):
            raise ValueError(f"Índice RES({n}) fora dos limites. Resultados disponíveis: {len(parser.resultados)}")
        # Ajusta o índice para acessar o N-ésimo resultado anterior
        # n=0 retorna o último resultado, n=1 retorna o penúltimo, etc.
        result = parser.resultados[-(n+1)]
        parser.resultados.append(result)
        return result

def print_ast_text(node, indent=0):
    prefix = "  " * indent
    print(f"{prefix}{node.__class__.__name__}", end="")
    if hasattr(node, 'value') and node.value is not None:
        print(f": {node.value}", end="")
    if node.type is not None:
        print(f" (Type: {node.type})")
    else:
        print()
    for child in getattr(node, 'children', []):
        print_ast_text(child, indent + 1)

def print_first_follow_sets():
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Conjuntos FIRST:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    first_sets = FirstSet()
    for non_terminal in sorted(first_sets.sets.keys()):
        print(f"{Fore.GREEN}FIRST({non_terminal}){Style.RESET_ALL} = {sorted(list(first_sets.get(non_terminal)))}")
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Conjuntos FOLLOW:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    follow_sets = FollowSet()
    for non_terminal in sorted(follow_sets.sets.keys()):
        print(f"{Fore.GREEN}FOLLOW({non_terminal}){Style.RESET_ALL} = {sorted(list(follow_sets.get(non_terminal)))}")
    print()

class Sequent:
    def __init__(self, context: Dict[str, Type], expr: str, type_: Type):
        self.context = context
        self.expr = expr
        self.type = type_

    def __str__(self):
        context_str = ", ".join(f"{var}: {type_}" for var, type_ in self.context.items())
        return f"{{{context_str}}} ⊢ {self.expr} : {self.type}"

class TypeInference:
    def __init__(self):
        self.derivations = []

    def _add_derivation(self, sequent: Sequent):
        self.derivations.append(sequent)

    def var_rule(self, var: str, context: Dict[str, Type]) -> Optional[Sequent]:
        if var in context:
            sequent = Sequent(context, var, context[var])
            self._add_derivation(sequent)
            return sequent
        return None

    def num_rule(self, num: Union[int, float]) -> Sequent:
        type_ = INT_TYPE if isinstance(num, int) else FLOAT_TYPE
        sequent = Sequent({}, str(num), type_)
        self._add_derivation(sequent)
        return sequent

    def binop_rule(self, op: str, e1_sequent: Sequent, e2_sequent: Sequent) -> Optional[Sequent]:
        context = {**e1_sequent.context, **e2_sequent.context} 

        if not e1_sequent.type.is_numeric or not e2_sequent.type.is_numeric:
            return None
        
        result_type = None
        if op in ['/', '|']:
            result_type = FLOAT_TYPE
        elif op in ['>', '<', '==', '!=', '<=', '>=']:
            result_type = BOOL_TYPE
        elif op == '^':
            if e2_sequent.type != INT_TYPE: return None
            result_type = FLOAT_TYPE
        elif e1_sequent.type == FLOAT_TYPE or e2_sequent.type == FLOAT_TYPE:
            result_type = FLOAT_TYPE
        else:
            result_type = INT_TYPE
        
        sequent = Sequent(context, f"({e1_sequent.expr} {e2_sequent.expr} {op})", result_type) # Changed expression format for RPN
        self._add_derivation(sequent)
        return sequent

    def if_rule(self, cond_sequent: Sequent, then_sequent: Sequent, else_sequent: Sequent) -> Optional[Sequent]:
        context = {**cond_sequent.context, **then_sequent.context, **else_sequent.context}

        if not (cond_sequent.type == BOOL_TYPE or cond_sequent.type.is_numeric):
            return None
        
        result_type = None
        if then_sequent.type.can_cast_to(else_sequent.type):
            result_type = else_sequent.type
        elif else_sequent.type.can_cast_to(then_sequent.type):
            result_type = then_sequent.type
        else:
            return None
            
        sequent = Sequent(context, f"(if {cond_sequent.expr} then {then_sequent.expr} else {else_sequent.expr})", result_type)
        self._add_derivation(sequent)
        return sequent

    def for_rule(self, var_name: str, start_sequent: Sequent, end_sequent: Sequent, body_sequent: Sequent) -> Optional[Sequent]:
        context = {**start_sequent.context, **end_sequent.context}
        
        if not (start_sequent.type == INT_TYPE and end_sequent.type == INT_TYPE):
            return None
        
        body_context_with_var = body_sequent.context.copy()
        body_context_with_var[var_name] = INT_TYPE
        
        sequent = Sequent(context, f"(for {var_name} in {start_sequent.expr} to {end_sequent.expr} do {body_sequent.expr})", body_sequent.type)
        self._add_derivation(sequent)
        return sequent

    def mem_store_rule(self, value_sequent: Sequent) -> Optional[Sequent]:
        if value_sequent.type.is_numeric:
            sequent = Sequent(value_sequent.context, f"({value_sequent.expr} MEM)", FLOAT_TYPE)
            self._add_derivation(sequent)
            return sequent
        return None

    def mem_retrieve_rule(self) -> Sequent:
        sequent = Sequent({}, "MEM", FLOAT_TYPE)
        self._add_derivation(sequent)
        return sequent

    def res_rule(self, offset_sequent: Sequent) -> Optional[Sequent]:
        # If (N RES) means return the N-th result (from the `resultados` list)
        if offset_sequent.type == INT_TYPE: # N must be an integer
            # The type of the result node will be FLOAT_TYPE (as results are stored as floats implicitly).
            sequent = Sequent(offset_sequent.context, f"({offset_sequent.expr} RES)", FLOAT_TYPE)
            self._add_derivation(sequent)
            return sequent
        return None

    def infer_type(self, node: ASTNode, context: Dict[str, Type] = None) -> Optional[Sequent]:
        if context is None:
            context = {}

        if isinstance(node, NumberNode):
            return self.num_rule(node.value)
        elif isinstance(node, IdentifierNode):
            if node.type:
                return Sequent(context, node.value, node.type)
            return self.var_rule(node.value, context)
        elif isinstance(node, BinaryOpNode):
            e1 = self.infer_type(node.children[0], context)
            e2 = self.infer_type(node.children[1], context)
            if e1 and e2:
                return self.binop_rule(node.value, e1, e2)
        elif isinstance(node, IfNode):
            cond = self.infer_type(node.children[0], context)
            then = self.infer_type(node.children[1], context)
            else_ = self.infer_type(node.children[2], context)
            if cond and then and else_:
                return self.if_rule(cond, then, else_)
        elif isinstance(node, ForNode):
            var_name = node.children[0].value
            start = self.infer_type(node.children[1], context)
            end = self.infer_type(node.children[2], context)
            
            body_context = context.copy()
            body_context[var_name] = INT_TYPE
            
            body = self.infer_type(node.children[3], body_context)
            if start and end and body:
                return self.for_rule(var_name, start, end, body)
        elif isinstance(node, MemoryStoreNode):
            value_sequent = self.infer_type(node.children[0], context)
            if value_sequent:
                return self.mem_store_rule(value_sequent)
        elif isinstance(node, MemoryRetrieveNode):
            return self.mem_retrieve_rule()
        elif isinstance(node, ResultNode):
            offset_sequent = self.infer_type(node.children[0], context)
            if offset_sequent:
                return self.res_rule(offset_sequent)
        elif isinstance(node, StringNode):
            return Sequent({}, f'"{node.value}"', STRING_TYPE)
        return None

def print_type_derivation(node: ASTNode):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Derivação de Tipos:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    
    type_inference = TypeInference()
    root_sequent = type_inference.infer_type(node)
    
    if root_sequent:
        print(f"{Fore.GREEN}Expressão: {root_sequent.expr}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Contexto Final: {root_sequent.context}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Tipo Inferido Final: {root_sequent.type}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Sequentes Gerados (ordem de aplicação):{Style.RESET_ALL}")
        for s in type_inference.derivations:
            print(f"  {s}")
    else:
        print(f"{Fore.RED}Não foi possível inferir o tipo para a expressão.{Style.RESET_ALL}")

# --- Utility Functions (These MUST be at the global scope, not inside main or any class) ---

def ast_to_dict(node):
    if isinstance(node, ASTNode):
        return {
            'type': node.__class__.__name__,
            'value': node.value,
            'children': [ast_to_dict(child) for child in node.children]
        }
    return node

def adicionar_rotinas_ieee754(file):
    """
    Adiciona as rotinas de manipulação IEEE 754 ao arquivo Assembly
    """
    file.write("""
;***********************************************************************************************
; Rotinas auxiliares (UART e Delay) - Posicionadas no início para melhor alcance de RCALL
;***********************************************************************************************

; Função para enviar um byte pela UART
uart_envia_byte:
    LDS R17, UCSR0A
    SBRS R17, 5
    RJMP uart_envia_byte
    STS UDR0, R16
    RET

; Função de delay em milissegundos
delay_ms:
    PUSH R20
    PUSH R21
    LDI R20, 100 ; Aumentado para 100 para um delay perceptível
delay_ms_outer:
    LDI R21, 255
delay_ms_inner:
    DEC R21
    BRNE delay_ms_inner
    DEC R20
    BRNE delay_ms_outer
    POP R21
    POP R20
    RET

;***********************************************************************************************
; Rotinas para manipulação de números IEEE 754 half-precision (16 bits)
;***********************************************************************************************

half_add:
    ; Empilhar registradores
    PUSH R20
    PUSH R21
    PUSH R22
    PUSH R23
    PUSH R24
    PUSH R25

    ; Extrair componentes do primeiro operando (a)
    MOV R20, R17         ; r20 = byte alto de a
    ANDI R20, 0x80       ; r20 = bit de sinal de a
    MOV R21, R17
    ANDI R21, 0x7C       ; r21 = 5 bits de expoente (parte alta) / bit de sinal (bit 8) e ignorando mantissa
    LSR R21
    LSR R21              ; r21 = expoente >> 2 / R21 > menos significativo a mantissa (2 zeros a esquerda)
    MOV R22, R17         ; começo da extração da mantissa
    ANDI R22, 0x03       ; r22 = 2 bits mais altos da mantissa
    LSL R22
    LSL R22
    LSL R22
    LSL R22
    LSL R22
    LSL R22              ; r22 = bits altos da mantissa deslocados
    OR R22, R16          ; r22:r16 = mantissa completa

    ; Extrair componentes do segundo operando (b)
    MOV R23, R19         ; r23 = byte alto de b
    ANDI R23, 0x80       ; r23 = bit de sinal de b
    MOV R24, R19
    ANDI R24, 0x7C       ; r24 = 5 bits de expoente (parte alta)
    LSR R24
    LSR R24              ; r24 = expoente >> 2
    MOV R25, R19
    ANDI R25, 0x03       ; r25 = 2 bits mais altos da mantissa
    LSL R25
    LSL R25
    LSL R25
    LSL R25
    LSL R25
    LSL R25              ; r25 = bits altos da mantissa deslocados
    OR R25, R18          ; r25:r18 = mantissa completa

    ; Alinhar expoentes
    CP R21, R24
    BREQ exponents_equal
    BRLO a_smaller
    
    ; Expoente de a é maior
    SUB R21, R24         ; Diferença entre expoentes
    ; Ajustar mantissa de b
    LSR R25
    JMP exponents_equal
    
a_smaller:
    ; Expoente de b é maior
    SUB R24, R21         ; Diferença entre expoentes
    ; Ajustar mantissa de a
    LSR R22
    
exponents_equal:
    ; Verificar sinais para soma/subtração
    CP R20, R23
    BREQ same_sign
    
    ; Sinais diferentes - realizar subtração
    JMP result_ready
    
same_sign:
    ; Sinais iguais - realizar soma
    ADD R16, R18         ; Somar bytes baixos da mantissa
    ADC R22, R25         ; Somar bytes altos da mantissa com carry
    
result_ready:
    ; Reconstruir o número IEEE 754 de 16 bits
    MOV R17, R20         ; Colocar bit de sinal
    
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET

half_subtract:
    ; Empilhar registradores
    PUSH R20
    PUSH R21
    PUSH R22
    PUSH R23
    PUSH R24
    PUSH R25

    ; Extrair componentes do primeiro operando (a)
    MOV R20, R17         ; r20 = byte alto de a
    ANDI R20, 0x80       ; r20 = bit de sinal de a
    MOV R21, R17
    ANDI R21, 0x7C       ; r21 = 5 bits de expoente (parte alta)
    LSR R21
    LSR R21              ; r21 = expoente >> 2
    MOV R22, R17
    ANDI R22, 0x03       ; r22 = 2 bits mais altos da mantissa
    LSL R22
    LSL R22
    LSL R22
    LSL R22
    LSL R22
    LSL R22              ; r22 = bits altos da mantissa deslocados
    OR R22, R16          ; r22:r16 = mantissa completa

    ; Extrair componentes do segundo operando (b)
    MOV R23, R19         ; r23 = byte alto de b
    ANDI R23, 0x80       ; r23 = bit de sinal de b
    MOV R24, R19
    ANDI R24, 0x7C       ; r24 = 5 bits de expoente (parte alta)
    LSR R24
    LSR R24              ; r24 = expoente >> 2
    MOV R25, R19
    ANDI R25, 0x03       ; r25 = 2 bits mais altos da mantissa
    LSL R25
    LSL R25
    LSL R25
    LSL R25
    LSL R25
    LSL R25              ; r25 = bits altos da mantissa deslocados
    OR R25, R18          ; r25:r18 = mantissa completa

    ; Inverter o sinal do segundo operando (transformar subtração em adição com sinal invertido)
    LDI R19, 0x80
    EOR R23, R19         ; Inverte o bit de sinal de b
    
    ; Alinhar expoentes
    CP R21, R24
    BREQ sub_exponents_equal
    BRLO sub_a_smaller
    
    ; Expoente de a é maior
    SUB R21, R24         ; Diferença entre expoentes
    ; Ajustar mantissa de b
    LSR R25
    DEC R21
    BRNE sub_exponents_equal
    JMP sub_exponents_equal
    
sub_a_smaller:
    ; Expoente de b é maior
    SUB R24, R21         ; Diferença entre expoentes
    ; Ajustar mantissa de a
    LSR R22
    DEC R24
    BRNE sub_a_smaller
    
sub_exponents_equal:
    ; Verificar sinais para soma/subtração
    CP R20, R23
    BREQ sub_same_sign
    
    ; Sinais diferentes - realizar subtração
    SUB R16, R18         ; Subtrair bytes baixos da mantissa
    SBC R22, R25         ; Subtrair bytes altos da mantissa com carry
    JMP sub_result_ready
    
sub_same_sign:
    ; Sinais iguais - realizar soma
    ADD R16, R18         ; Somar bytes baixos da mantissa
    ADC R22, R25         ; Somar bytes altos da mantissa com carry
    
sub_result_ready:
    ; Reconstruir o número IEEE 754 de 16 bits
    MOV R17, R20         ; Colocar bit de sinal
    
    ; Normalizar resultado se necessário
    SBRC R22, 7          ; Se bit 7 estiver setado, ajustar expoente
    INC R21              ; Incrementar expoente
    
    ; Inserir expoente
    LSL R21
    LSL R21              ; Deslocar expoente
    ANDI R17, 0x83       ; Manter sinal e 2 bits altos da mantissa
    ANDI R21, 0x7C       ; Manter apenas os 5 bits do expoente
    OR R17, R21          ; Combinar sinal + expoente + bits altos da mantissa
    
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET

half_multiply:
    ; Empilhar registradores
    PUSH R20
    PUSH R21
    PUSH R22
    PUSH R23
    PUSH R24
    PUSH R25
    
    ; Extrair sinal (XOR dos bits de sinal)
    MOV R20, R17         ; Byte alto de a
    ANDI R20, 0x80       ; Bit de sinal de a
    MOV R21, R19         ; Byte alto de b
    ANDI R21, 0x80       ; Bit de sinal de b
    EOR R20, R21         ; r20 = sinal do resultado
    
    ; Extrair expoentes
    MOV R21, R17
    ANDI R21, 0x7C       ; 5 bits de expoente de a
    LSR R21
    LSR R21              ; r21 = expoente normalizado
    
    MOV R22, R19
    ANDI R22, 0x7C       ; 5 bits de expoente de b
    LSR R22
    LSR R22              ; r22 = expoente normalizado
    
    ; Somar expoentes e subtrair bias (15)
    ADD R21, R22
    SUBI R21, 15         ; r21 = expoente final
    
    ; Reconstruir o número IEEE 754 de 16 bits
    MOV R17, R20         ; Colocar bit de sinal
    
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET

half_divide:
    ; Empilhar registradores
    PUSH R20
    PUSH R21
    PUSH R22
    PUSH R23
    PUSH R24
    PUSH R25
    
    ; Verificar divisão por zero
    MOV R20, R18
    OR R20, R19
    BREQ half_div_by_zero
    
    ; Extrair sinal (XOR dos bits de sinal)
    MOV R20, R17         ; Byte alto de a
    ANDI R20, 0x80       ; Bit de sinal de a
    MOV R21, R19         ; Byte alto de b
    ANDI R21, 0x80       ; Bit de sinal de b
    EOR R20, R21         ; r20 = sinal do resultado
    
    ; Extrair expoentes
    MOV R21, R17
    ANDI R21, 0x7C       ; 5 bits de expoente de a
    LSR R21
    LSR R21              ; r21 = expoente normalizado
    
    MOV R22, R19
    ANDI R22, 0x7C       ; 5 bits de expoente de b
    LSR R22
    LSR R22              ; r22 = expoente normalizado
    
    ; Calcular expoente do resultado: exp_a - exp_b + bias(15)
    SUB R21, R22
    SUBI R21, -15        ; Adicionar bias (usando subtração negativa)
    
    ; Extrair mantissas
    MOV R22, R17
    ANDI R22, 0x03       ; 2 bits mais altos da mantissa de a
    LSL R22
    LSL R22
    LSL R22
    LSL R22
    LSL R22
    LSL R22              ; r22 = bits altos da mantissa deslocados
    OR R22, R16          ; r22:r16 = mantissa completa de a
    
    MOV R23, R19
    ANDI R23, 0x03       ; 2 bits mais altos da mantissa de b
    LSL R23
    LSL R23
    LSL R23
    LSL R23
    LSL R23
    LSL R23              ; r23 = bits altos da mantissa deslocados
    OR R23, R18          ; r23:r18 = mantissa completa de b
    
    ; Adicionar bit implícito para mantissas normalizadas
    ORI R22, 0x40        ; Adicionar bit implícito à mantissa de a
    ORI R23, 0x40        ; Adicionar bit implícito à mantissa de b
    
    ; Realizar a divisão da mantissa (simplificada)
    ; Normalmente, isso seria feito com uma rotina de divisão completa
    ; Mas para simplificar, assumimos que a mantissa do resultado é aproximada
    
    ; Reconstruir o número IEEE 754 de 16 bits
    MOV R17, R20         ; Colocar bit de sinal
    
    ; Inserir expoente
    LSL R21
    LSL R21              ; Deslocar expoente
    ANDI R17, 0x83       ; Manter sinal e 2 bits altos da mantissa
    ANDI R21, 0x7C       ; Manter apenas os 5 bits do expoente
    OR R17, R21          ; Combinar sinal + expoente + bits altos da mantissa
    
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET
    
half_div_by_zero:
    ; Extrair o sinal do numerador (a)
    MOV R20, R17
    ANDI R20, 0x80       ; Bit de sinal de a
    
    ; Gerar infinito com o sinal apropriado
    LDI R16, 0x00        ; Byte baixo para infinito
    LDI R17, 0x7C        ; Byte alto para infinito positivo
    OR R17, R20          ; Aplicar sinal ao infinito
    
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET

half_power:
    ; Empilhar registradores
    PUSH R20
    PUSH R21
    PUSH R22
    PUSH R23
    PUSH R24
    PUSH R25
    
    ; Verificar casos especiais
    ; Caso 1: Se expoente for 0, retornar 1.0
    MOV R20, R18
    OR R20, R19
    BREQ power_one       ; Se expoente for zero, resultado é 1.0
    
    ; Caso 2: Se base for 1.0, retornar 1.0
    LDI R20, 0x00
    CP R16, R20
    BRNE check_base_neg
    LDI R20, 0x3C        ; 1.0 em half-precision
    CP R17, R20
    BREQ power_one       ; Se base for 1.0, resultado é 1.0
    
check_base_neg:
    ; Caso 3: Se base for negativa, verificar se expoente é inteiro
    MOV R20, R17
    ANDI R20, 0x80
    BREQ base_positive   ; Se base for positiva, prosseguir normalmente
    
    ; Base é negativa, verificar se expoente é inteiro
    ; Para simplificar, vamos retornar NaN para base negativa
    JMP power_nan
    
base_positive:
    ; Implementação simplificada: para potência, usamos logaritmo
    ; ln(a^b) = b * ln(a), depois exp()
    ; Como isso requer funções transcendentais complexas,
    ; vamos implementar apenas casos especiais comuns
    
    ; Caso especial: Se expoente for 0.5, calcular raiz quadrada
    LDI R20, 0x00
    CP R18, R20
    BRNE power_approx
    LDI R20, 0x38        ; 0.5 em half-precision
    CP R19, R20
    BRNE power_approx
    
    ; Calcular raiz quadrada (aproximação)
    ; Para simplificar, dividimos o expoente por 2
    MOV R21, R17
    ANDI R21, 0x7C       ; Expoente da base
    LSR R21              ; Dividir expoente por 2
    ANDI R17, 0x83       ; Manter sinal e parte da mantissa
    OR R17, R21          ; Recombinar
    JMP power_done
    
power_approx:
    ; Implementação muito simplificada para outros casos
    ; Ajuste de expoente aproximado para operações de potência
    ; Em uma implementação real, um algoritmo mais complexo seria necessário
    
power_done:
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET
    
power_one:
    ; Retornar 1.0
    LDI R16, 0x00
    LDI R17, 0x3C        ; 1.0 em half-precision
    
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET
    
power_nan:
    ; Retornar NaN
    LDI R16, 0x00
    LDI R17, 0x7E        ; NaN em half-precision
    
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET

half_modulo:
    ; Empilhar registradores
    PUSH R20
    PUSH R21
    PUSH R22
    PUSH R23
    PUSH R24
    PUSH R25
    
    ; Verificar divisão por zero
    MOV R20, R18
    OR R20, R19
    BREQ mod_by_zero
    
    ; Extrair componentes do primeiro operando (a)
    MOV R20, R17         ; r20 = byte alto de a
    ANDI R20, 0x80       ; r20 = bit de sinal de a
    
    ; Extrair expoentes
    MOV R21, R17
    ANDI R21, 0x7C       ; 5 bits de expoente de a
    MOV R22, R19
    ANDI R22, 0x7C       ; 5 bits de expoente de b
    
    ; Verificar se b é maior que a
    CP R21, R22
    BRLO mod_a_smaller   ; Se expoente de a for menor, resultado é a
    
    ; Implementação simplificada de módulo
    ; Em uma implementação real, precisaríamos realizar a divisão,
    ; truncar para obter o quociente, multiplicar pelo divisor,
    ; e subtrair do dividendo
    
    ; Para simplificar, vamos retornar um valor aproximado baseado
    ; na comparação dos expoentes
    
mod_a_smaller:
    ; Se a < b, resultado do módulo é a
    MOV R16, R16
    MOV R17, R17
    JMP mod_done
    
mod_done:
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET
    
mod_by_zero:
    ; Retornar NaN
    LDI R16, 0x00
    LDI R17, 0x7E        ; NaN em half-precision
    
    ; Restaurar registradores
    POP R25
    POP R24
    POP R23
    POP R22
    POP R21
    POP R20
    RET

integer_divide:
    ; Empilhar registradores
    PUSH R20
    PUSH R21
    PUSH R22
    PUSH R23
    
    ; Verificar divisão por zero
    MOV R20, R18
    OR R20, R19
    BREQ div_by_zero
    
    ; Salvar sinal
    MOV R20, R16
    EOR R20, R18         ; XOR para determinar o sinal do resultado
    ANDI R20, 0x80       ; Apenas o bit de sinal
    
    ; Resultado da divisão inteira em r16:r17
    ; Restaurar o sinal em r17
    ANDI R17, 0x7F       ; Limpar bit de sinal
    OR R17, r20          ; Aplicar bit de sinal
    
    ; Restaurar registradores
    POP R23
    POP R22
    POP R21
    POP R20
    RET
    
div_by_zero:
    ; Tratar divisão por zero
    LDI R16, 0xFF       ; Indicar erro
    LDI R17, 0xFF
    
    ; Restaurar registradores
    POP R23
    POP R22
    POP R21
    POP R20
    RET
""")

def main():
    if len(sys.argv) != 2:
        print(f"{Fore.RED}Uso: python compilador.py arquivo.txt{Style.RESET_ALL}")
        sys.exit(1)

    print_first_follow_sets()
    
    parsing_table = ParsingTable()
    parsing_table.print_table()
    
    arquivo = sys.argv[1]
    with open(arquivo, 'r') as f:
        linhas = []
        for l in f:
            l = l.split(';')[0].strip()
            if l:
                linhas.append(l)
    
    memoria = 0.0
    resultados = []
    
    with open('calculadora.asm', 'w') as asm:
        asm.write("""; Calculadora RPN - Código Assembly para ATmega328P (IEEE 754 Half-precision 16 bits)
; Alunos: Gabriel Martins Vicente, Javier Agustin Aranibar González, Matheus Paul Lopuch, Rafael Bonfim Zacco
;***********************************************************************************************
.equ SPH, 0x3E    ; Stack Pointer High
.equ SPL, 0x3D    ; Stack Pointer Low
.equ UBRR0L, 0xC4 ; Baud Rate Register Low
.equ UBRR0H, 0xC5 ; Baud Rate Register High
.equ UCSR0A, 0xC0 ; Control and Status Register A: Usado para verificar o status da UART (bit 5: UDRE0)
.equ UCSR0B, 0xC1 ; Control and Status Register B: Habilita transmissor (bit TXEN0)
.equ UCSR0C, 0xC2 ; Control and Status Register C: Configura formato de dados (UCSZ00 e UCSZ01)
.equ UDR0, 0xC6   ; Registrador de dados para a UART0
; Fórmula para definir o Universal Boud Rate Register (UBRR): UBRR = Fcpu / (16 * Baud Rate) -1
; Para o ATmega328P seria: 16MHz / (16 * 9600) - 1 = 103
;***********************************************************************************************

.ORG 0x0000
    RJMP reset
    
reset:
    ; Configurar stack pointer
    LDI R16, 0x08
    OUT SPH, r16
    LDI R16, 0xFF
    OUT SPL, r16
        
    ; Configurar UART
    LDI R16, 103
    STS UBRR0L, r16
    LDI R16, 0
    STS UBRR0H, r16
    LDI R16, (1<<3) ; TXEN0 bit 3
    STS UCSR0B, r16
    LDI R16, (3<<1) ; UCSZ00 e UCSZ01 bits 1,2 (8 bits de dados)
    STS UCSR0C, r16
        
    ; Delay inicial para estabilizar o sistema
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    
    ; Enviar mensagem inicial
    LDI R16, 'C'
    RCALL uart_envia_byte
    LDI R16, 'a'
    RCALL uart_envia_byte
    LDI R16, 'l'
    RCALL uart_envia_byte
    LDI R16, 'c'
    RCALL uart_envia_byte
    LDI R16, 'u'
    RCALL uart_envia_byte
    LDI R16, 'l'
    RCALL uart_envia_byte
    LDI R16, 'a'
    RCALL uart_envia_byte
    LDI R16, 'd'
    RCALL uart_envia_byte
    LDI R16, 'o'
    RCALL uart_envia_byte
    LDI R16, 'r'
    RCALL uart_envia_byte
    LDI R16, 'a'
    RCALL uart_envia_byte
    LDI R16, ' '
    RCALL uart_envia_byte
    LDI R16, 'R'
    RCALL uart_envia_byte
    LDI R16, 'P'
    RCALL uart_envia_byte
    LDI R16, 'N'
    RCALL uart_envia_byte
    LDI R16, ':'
    RCALL uart_envia_byte
    LDI R16, 13  ; CR
    RCALL uart_envia_byte
    LDI R16, 10  ; LF
    RCALL uart_envia_byte
    LDI R16, 13  ; CR (linha extra para espaçamento visual)
    RCALL uart_envia_byte
    LDI R16, 10  ; LF
    RCALL uart_envia_byte
    
main:
""")
        
        for i, expr in enumerate(linhas):
            print(f"\n{Fore.CYAN}{Style.BRIGHT}Expressão {i+1}:{Style.RESET_ALL} {expr}")
            tokens_lex = lexer(expr, i+1)
            if tokens_lex:
                print(f"  {Fore.YELLOW}Tokens:{Style.RESET_ALL}")
                for t in tokens_lex:
                    print(f"    {t}")
            try:
                parser = Parser(tokens_lex, memoria, resultados)
                parser.set_allow_casting(True)
                ast = parser.parse()
                
                try:
                    ast.type = ast.check_types()
                    print_type_derivation(ast)
                except ValueError as e:
                    print(f"  {Fore.RED}Erro semântico: {e}{Style.RESET_ALL}")
                    continue
                
                result = parser.eval_ast(ast)
                memoria = parser.memoria
                
                # A lógica de adição de resultados já está no eval_ast para a maioria dos nós
                # E o controle de duplicação para BinaryOpNode (comparações) está lá também.
                # Não é mais necessário um append extra aqui, pois eval_ast já lida.
                # Apenas para garantir que o resultado final da expressão seja armazenado se não for um if/for
                if not isinstance(ast, MemoryStoreNode) and not (isinstance(ast, BinaryOpNode) and ast.type == BOOL_TYPE):
                    if not (isinstance(ast, ForNode) or isinstance(ast, IfNode)) or isinstance(result, (int, float)):
                        # Se for For/If, só adiciona se o resultado for numérico
                        pass # A adição já acontece em eval_ast para estes casos também.


                ieee_hex = float_to_ieee754(float(result))
                print(f"  {Fore.GREEN}Resultado: {result} [IEEE754: {ieee_hex}]{Style.RESET_ALL}")
                
                tipo_str = '?'
                if isinstance(ast, IfNode):
                    # Para if, o tipo é inferido dos branches
                    cond_val = parser.eval_ast(ast.children[0]) # Reavalia a condição para determinar o branch
                    if bool(cond_val):
                        final_branch_type = ast.children[1].type
                    else:
                        final_branch_type = ast.children[2].type

                    if final_branch_type == INT_TYPE:
                        tipo_str = 'i'
                    elif final_branch_type == FLOAT_TYPE:
                        tipo_str = 'f'
                    elif final_branch_type == BOOL_TYPE: # Se o resultado do if é booleano, trata como tal
                        tipo_str = 'b'
                    else:
                        tipo_str = 'u' # Tipo desconhecido ou outro
                elif isinstance(ast, ForNode):
                    # Para loop for, o tipo é inferido do corpo
                    if ast.type == INT_TYPE:
                        tipo_str = 'i'
                    elif ast.type == FLOAT_TYPE:
                        tipo_str = 'f'
                    elif ast.type == BOOL_TYPE:
                        tipo_str = 'b'
                    else:
                        tipo_str = 'u'
                elif isinstance(ast, BinaryOpNode) and ast.type == BOOL_TYPE:
                    tipo_str = 'b' # Resultado booleano para comparações
                else:
                    if ast.type == INT_TYPE:
                        tipo_str = 'i'
                    elif ast.type == FLOAT_TYPE:
                        tipo_str = 'f'
                    else:
                        tipo_str = 'u' # Catch-all para outros tipos de nó
                
                dot = ast.visualize()
                dot.render(f'ast_{i+1}', format='png', cleanup=True)
                print(f"  {Fore.YELLOW}AST gerada em ast_{i+1}.png{Style.RESET_ALL}")
                print(f"\n{Fore.CYAN}Árvore Sintática (texto):{Style.RESET_ALL}")
                print_ast_text(ast)
                print(f"\n{Fore.CYAN}Árvore Sintática (JSON):{Style.RESET_ALL}")
                print(json.dumps(ast_to_dict(ast), indent=2, ensure_ascii=False))
                
                escrever_serial(asm, expr, result, ieee_hex, tipo_str)
                
            except Exception as e:
                print(f"  {Fore.RED}Erro ao processar expressão: {e}{Style.RESET_ALL}")
                continue
        asm.write("""
    ; Loop infinito para o final do programa
loop_end:
    RJMP loop_end
""") # Ajustado para loop_end para clareza
        # Adicionar rotinas IEEE 754 e de delay no final do arquivo ASM
        adicionar_rotinas_ieee754(asm) # Chamado aqui para garantir que as rotinas sejam definidas UMA VEZ
    
    print(f"\n{Fore.GREEN}Arquivo calculadora.asm gerado com sucesso!{Style.RESET_ALL}")

# --- Entry Point for the script ---
if __name__ == "__main__":
    main()
