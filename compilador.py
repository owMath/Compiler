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
        
        for op_val in OPERATORS:
            self.table['OPERATOR']['operador'] = [op_val]
        
        self.table['NUM']['número'] = ['INT']
    
    def get_production(self, non_terminal, terminal_type_or_value):
        if terminal_type_or_value == 'número':
            return self.table.get(non_terminal, {}).get('número')
        elif terminal_type_or_value == 'identificador':
            return self.table.get(non_terminal, {}).get('identificador')
        elif terminal_type_or_value in KEYWORDS:
            return self.table.get(non_terminal, {}).get(terminal_type_or_value)
        elif terminal_type_or_value in OPERATORS:
            return self.table.get(non_terminal, {}).get('operador')
        
        return self.table.get(non_terminal, {}).get(terminal_type_or_value)
    
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
        if f == 0.0: return '0x0000'
        if f < 0.0 and f == 0.0: return '0x8000'
        if f == float('inf'): return '0x7C00'
        if f == float('-inf'): return '0xFC00'
        if math.isnan(f): return '0x7E00'
        
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
                self.resultados.append(result)
                return result
            elif isinstance(node, IfNode):
                cond = self.eval_ast(node.children[0])
                if not (isinstance(cond, bool) or isinstance(cond, (int, float))):
                    raise ValueError(f"Condição do 'if' deve ser booleana ou numérica, mas encontrou tipo {type(cond).__name__}")
                result = self.eval_ast(node.children[1] if bool(cond) else node.children[2])
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
                self.resultados.append(self.memoria)
                return self.memoria
            elif isinstance(node, MemoryRetrieveNode):
                result = self.memoria
                self.resultados.append(result)
                return result
            elif isinstance(node, ResultNode):
                n = int(self.eval_ast(node.children[0]))
                if n < 0 or n >= len(self.resultados):
                    raise ValueError(f"Índice RES({n}) fora dos limites. Resultados disponíveis: {len(self.resultados)}")
                result = self.resultados[-(n)]  # retorna a função RES
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
                return a ** b
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
            if self.peek_next() and self.peek_next().value == ')':
                self.eat('palavra-chave', 'MEM')
                self.eat('parêntese', ')')
                return MemoryRetrieveNode()

        elif t.type == 'palavra-chave' and t.value == 'RES':
            if self.peek_next() and self.peek_next().value == ')':
                self.eat('palavra-chave', 'RES')
                self.eat('parêntese', ')')
                # (RES) meaning: last result (N=0)
                return ResultNode(NumberNode(0)) 

        first_expr_node = self.parse() 
        next_token_after_first_expr = self.at()

        if next_token_after_first_expr is None:
             raise ValueError("Expressão incompleta. Esperado mais tokens após o primeiro operando.")

        if next_token_after_first_expr.type == 'palavra-chave' and next_token_after_first_expr.value == 'MEM':
            self.eat('palavra-chave', 'MEM')
            self.eat('parêntese', ')')
            mem_store_node = MemoryStoreNode()
            mem_store_node.add_child(first_expr_node)
            return mem_store_node
        
        elif next_token_after_first_expr.type == 'palavra-chave' and next_token_after_first_expr.value == 'RES':
            self.eat('palavra-chave', 'RES')
            self.eat('parêntese', ')')
            if not isinstance(first_expr_node, NumberNode):
                raise ValueError(f"Esperado um número para N em (N RES), mas encontrou {type(first_expr_node).__name__} na linha {first_expr_node.line}, coluna {first_expr_node.col}")
            # This is the crucial change: ResultNode now just holds N
            res_node = ResultNode(first_expr_node) 
            return res_node
        
        if next_token_after_first_expr is None or (not self.check_first('EXPR', next_token_after_first_expr) and next_token_after_first_expr.type != 'operador'):
             raise ValueError(f'Token inesperado: {next_token_after_first_expr}. Esperado um dos tokens em FIRST(EXPR) para o segundo operando ou um operador na linha {next_token_after_first_expr.line}, coluna {next_token_after_first_expr.col}.')

        b = self.parse()
        op_token = self.eat('operador')
        
        if op_token is None or not self.check_first('OPERATOR', op_token):
            raise ValueError(f'Operador inesperado: {op_token}. Esperado um dos operadores em FIRST(OPERATOR): {self.first_sets.get("OPERATOR")}')
        
        bin_op = BinaryOpNode(op_token.value)
        bin_op.add_child(first_expr_node)
        bin_op.add_child(b)
        self.eat('parêntese', ')')
        return bin_op

def escrever_serial(asm, operacao, resultado, ieee_hex, tipo_str):
    texto = f"{operacao} = {resultado} [IEEE754: {ieee_hex}] {tipo_str}"
    asm.write(f"    ; {texto}\n")
    for c in texto:
        asm.write(f"    LDI R16, 0x{ord(c):02X}\n    RCALL uart_envia_byte\n")
    asm.write("    LDI R16, 13\n    RCALL uart_envia_byte\n    LDI R16, 10\n    RCALL uart_envia_byte\n")

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

    def check_types(self):
        if self.type is None:
            self.type = INT_TYPE
        if not isinstance(self.type, Type):
            raise ValueError(f"Invalid type object for identifier '{self.value}': {self.type}")
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
        self.add_child(offset_node)
        self.type = None

    def check_types(self):
        if len(self.children) != 1:
            raise ValueError("Result node requires exactly one offset (N)")
        
        offset_type = self.children[0].check_types()
        self.type = offset_type # Type of RES is the type of N
        return self.type

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
        
        sequent = Sequent(context, f"({e1_sequent.expr} {op} {e2_sequent.expr})", result_type)
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
            # This rule in TypeInference assumes we can determine the type of the N-th result.
            # For this simplified type system, we'll assume it's float or the type of N.
            # It's better to stick to the actual eval_ast behavior.
            # So, the type of the result node will be FLOAT_TYPE (as results are stored as floats implicitly).
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
    """Converts an AST node to a dictionary (for JSON serialization)."""
    return {
        'type': node.__class__.__name__,
        'value': node.value,
        'node_type': str(node.type) if node.type else None, # Add inferred type to JSON
        'children': [ast_to_dict(child) for child in getattr(node, 'children', [])]
    }

# This is a placeholder/legacy function. 
# The eval_op method is now directly part of the Parser class.
def patch_eval_op():
    pass

# --- Main Function (This MUST be at the global scope, no indentation) ---

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
        asm.write("""; Assembly gerado para exibir resultados de expressões RPN\n.ORG 0x0000\n    RJMP main\nmain:\n""")
        asm.write("""
    ; Definições dos bits do UART
    .equ UCSR0B, 0xC1
    .equ UCSR0C, 0xC2
    .equ UBRR0L, 0xC4
    .equ UBRR0H, 0xC5
    .equ UDR0, 0xC6
    .equ UCSR0A, 0xC0
    .equ TXEN0, 3
    .equ UCSZ00, 1
    .equ UCSZ01, 2
    .equ UDRE0, 5

    ; Configurar UART (9600 baud com F_CPU=16MHz)
    ; UBRR0 = (F_CPU / 16 / BAUD) - 1 = (16000000 / 16 / 9600) - 1 = 103
    LDI r16, 103
    STS UBRR0L, r16   ; UBRR0L
    LDI r16, 0
    STS UBRR0H, r16   ; UBRR0H
    LDI r16, (1<<TXEN0) ; Habilita o transmissor (TXEN0 bit 3 de UCSR0B)
    STS UCSR0B, r16   ; UCSR0B
    LDI r16, (3<<UCSZ00) ; Configura 8 bits de dados, 1 bit de parada (UCSZ00 e UCSZ01 bits 1,2 de UCSR0C)
    STS UCSR0C, r16   ; UCSR0C
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
                
                # Only add to results if it's an expression that produces a value
                # and is not explicitly a MemoryStoreNode.
                # If (N RES) and (RES) retrieve results, they *do* produce a value
                # that should be added to the results list.
                # So, only exclude MemoryStoreNode.
                if not isinstance(ast, MemoryStoreNode):
                    resultados.append(result)
                
                ieee_hex = float_to_ieee754(float(result))
                print(f"  {Fore.GREEN}Resultado: {result} [IEEE754: {ieee_hex}]{Style.RESET_ALL}")
                
                # Usa o tipo inferido da AST para decidir se é int ou float
                if ast.type == INT_TYPE:
                    tipo_str = 'i'
                elif ast.type == FLOAT_TYPE:
                    tipo_str = 'f'
                else:
                    tipo_str = '?'
                
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
        asm.write("""loop:\n    RJMP loop\n\n; Rotina UART para ATmega328P\nuart_envia_byte:\n    ; Aguarda o registrador de buffer de transmissão estar vazio (UDRE0 bit 5 de UCSR0A)
    lds r17, UCSR0A   ; UCSR0A
    sbrs r17, UDRE0     ; Testa o bit UDRE0
    rjmp uart_envia_byte ; Se não vazio, pula de volta
    sts UDR0, r16   ; UDR0 (Envia o byte em R16)
    ret\n""")
    print(f"\n{Fore.GREEN}Arquivo calculadora.asm gerado com sucesso!{Style.RESET_ALL}")

# --- Entry Point for the script ---
if __name__ == "__main__":
    main()
