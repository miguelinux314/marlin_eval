#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare MarlinBaseTree chapters and other dictionaries
"""
__author__ = "Miguel Hernández Cabronero <mhernandez@deic.uab.cat>"
__date__ = "04/07/2019"

import os
import sys
import heapq
import time
import numpy as np
import math
import scipy.stats
import itertools
import pickle
import shutil
import collections
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pinvoker
import pandas as pd

from plotdata import LineData

sys.setrecursionlimit(2 ** 16)

# --- Begin configuration

## Simulation parameters
minimal_run = False  # If True, a reduced simulation is conducted
symbol_count_exponent = 4
symbol_count = 2 ** symbol_count_exponent  # Number of symbols in the source
test_input_length = 10 ** 4 if minimal_run else 2 ** 17  # Length of the synthetic sequences
# minimum, maximum entropy (relative to the theoretical maximum) and number of distributions to evaluate
min_max_count_tested_e_fractions = (0, 1, 40)
# Fixed seed to make results reproducible
seed = (0xbadc0ffe3 + 0x1fa14fe1) % (2 ** 32 - 1)
show_progress = False
exit_on_error = True
multiprocess_verbose = True
min_shown_efficiency = 0.5

generate_fig1 = True
generate_fig2 = True
generate_fig3 = True

# Everything else is a parameter
efficiency_common_columns = [
    "cls", "codec", "codec_path", "is_raw",
    "source", "source_len", "source_affix", "source_entropy",
    "input_len", "input_entropy", "bps"]

## Process count and prioritization
process_limit = None  # None to use all, 1 to use a single process
# process_limit = 1  # None to use all, 1 to use a single process
os.nice(10)

# Simulation paths
efficiency_results_path = "efficiency_results.csv"
overwrite_efficiency_results = False

input_plot_dir = "plots/input"
prebuilt_dir = "prebuilt"
os.makedirs(prebuilt_dir, exist_ok=True)
synthetic_inputs_dir = "synthetic_inputs"
# If False, codecs can be loaded from their .zip
overwrite_prebuilt_codecs = False
efficiency_plot_dir = "plots/efficiency"

# --- End configuration

os.makedirs(input_plot_dir, exist_ok=True)
os.makedirs(prebuilt_dir, exist_ok=True)
os.makedirs(synthetic_inputs_dir, exist_ok=True)
os.makedirs(efficiency_plot_dir, exist_ok=True)


def input_entropy(input):
    counter = collections.Counter(input)
    assert len(input) == sum(counter.values())
    symbols = sorted(counter.keys(), reverse=True, key=lambda symbol: symbol.p)
    pdf = [counter[symbol] / len(input) for symbol in symbols]
    return sum(-p * math.log2(p) if p > 0 else 0 for p in pdf)


class Symbol:
    def __init__(self, label, p):
        self.label = label
        self.p = p

    def __repr__(self):
        return str(self.label)

    # def __str__(self):
    #     return f"[Symbol label={self.label} p={self.p}]"

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return self.label == other.label


class Source:
    min_symbol_probability = 1e-10

    def __init__(self, symbols, affix=""):
        if symbols:
            assert round(sum(s.p for s in symbols), 10) == 1, symbols
            assert all(s.p >= 0 for s in symbols)
        self.symbols = list(sorted(symbols, key=lambda s: -s.p))
        self.affix = affix

    @property
    def name(self):
        name = f"{self.__class__.__name__}"
        if self.affix:
            name += f"_{self.affix}"
        name += f"_{len(self.symbols)}symbols_entropy{self.entropy:.7f}"
        return name

    @staticmethod
    def get_meta(source, symbols_per_meta):
        """Build a meta-source from source.
        :param source: source from which the meta-source is to be created
        :param symbols_per_meta: meta_symbols represent at most symbols_per_meta
          symbols
        :return: (
          # the created meta source
          meta_source,
          # dictionary indexed by metasymbol, entries are list of contained symbols
          metasymbol_to_symbols,
          # dictionary indexed by (a source's) symbol, entries are indices
          # of symbol in metasymbol_to_symbol[symbol]
          symbol_to_metasymbol_remainder:
          )
        """
        assert symbols_per_meta > 0
        meta_symbols = []
        metasymbol_to_symbols = {}
        symbol_to_metasymbol_remainder = {}
        for i in range(math.ceil(len(source.symbols) / symbols_per_meta)):
            symbols_in_meta = source.symbols[i * (symbols_per_meta):(i + 1) * (symbols_per_meta)]
            meta_symbols.append(Symbol(label=f"Q{i}", p=sum(s.p for s in symbols_in_meta)))
            symbol_to_metasymbol_remainder.update({symbol: (meta_symbols[-1], remainder)
                                                   for remainder, symbol in enumerate(symbols_in_meta)})
            metasymbol_to_symbols[meta_symbols[-1]] = list(symbols_in_meta)

        # Sanity check
        for symbol in source.symbols:
            meta_symbol, remainder = symbol_to_metasymbol_remainder[symbol]
            assert metasymbol_to_symbols[meta_symbol][remainder] is symbol

        meta_source = Source(symbols=meta_symbols, affix=f"meta-{source.name}")

        if abs(sum(source.probabilities) - 1) > 1e-10:
            print(f"[WARNING:Source.get_meta()] input source has psum = {sum(source.probabilities):12} != 1")
        assert abs(sum(meta_source.probabilities) - sum(source.probabilities)) < 1e-10

        return meta_source, metasymbol_to_symbols, symbol_to_metasymbol_remainder

    @staticmethod
    def get_laplacian(symbol_count, max_entropy_fraction):
        """
        :param max_entropy_fraction: fraction of the maximum entropy (ie, the entropy
          obtained for a uniform distribution) for the returned Source
        :return a Laplacian source with symbol_count symbols
        """
        assert 0 <= max_entropy_fraction <= 1
        assert symbol_count > 0
        max_entropy = math.log2(symbol_count)
        entropy_fraction = -1

        min_b = 1e-20
        max_b = 1e10
        while abs(max_entropy_fraction - entropy_fraction) > 1e-10:
            mid_b = 0.5 * (min_b + max_b)
            R = 100
            pdf = [0] * symbol_count
            for i in range(R * symbol_count):
                pdf[i % symbol_count] += max(0, math.exp(-(i) / mid_b))
            # pdf = [max(Source.min_symbol_probability, p) for p in pdf]
            pdf = [v if v > Source.min_symbol_probability else 0 for v in pdf]
            pdf = [p / sum(pdf) for p in pdf]
            entropy_fraction = sum(-p * math.log2(p) for p in pdf if p > 0) / max_entropy
            if entropy_fraction > 1 - 1e-6:
                pdf = [1 / len(pdf)] * len(pdf)
                entropy_fraction = 1
            if entropy_fraction > max_entropy_fraction:
                max_b = mid_b
            else:
                min_b = mid_b
            if abs(min_b - max_b) < 1e-10:
                print("[watch] entropy_fraction = {}".format(entropy_fraction))
                print("[watch] min_b = {}".format(min_b))
                print("[watch] max_b = {}".format(max_b))
                print("[watch] min(pdf) - max(pdf) = {}".format(min(pdf) - max(pdf)))
                raise RuntimeError(f"Source probabilitites didn't converge :: "
                                   f"symbol_count={symbol_count} "
                                   f"max_entropy_fraction={max_entropy_fraction}")

        source = Source([Symbol(label=i, p=p) for i, p in enumerate(pdf)])

        source.affix = "Laplacian"
        assert (max_entropy_fraction - source.entropy / max_entropy) <= 1e-10, (
            max_entropy_fraction, source.entropy / max_entropy)
        return source

    @property
    def entropy(self):
        """:return: the binary source entropy given the symbol distribution
        """
        assert abs(sum(self.probabilities) - 1) < 1e-5, abs(1 - sum(self.probabilities))
        return sum(-math.log2(s.p) * s.p if s.p else 0 for s in self.symbols)

    def generate_symbols(self, count):
        """Get a list of Symbols using the given probability distribution
        """
        ideal_symbols = list(itertools.chain(*([symbol] * math.ceil(symbol.p * count)
                                               for symbol in self.symbols)))
        ideal_symbols += [self.symbols[0]] * (count - len(ideal_symbols))
        ideal_symbols = ideal_symbols[len(ideal_symbols) - count:]
        np.random.seed(seed)
        np.random.shuffle(ideal_symbols)
        assert abs(len(ideal_symbols) - count) < len(self.symbols), (len(ideal_symbols), count)
        if count > 10 ** 4:
            assert abs(self.entropy - input_entropy(ideal_symbols)) < 1e-2, \
                (self.entropy, input_entropy(ideal_symbols), self.entropy - input_entropy(ideal_symbols))
        return ideal_symbols

    @property
    def probabilities(self):
        return [s.p for s in self.symbols]

    @property
    def labels(self):
        return [s.label for s in self.symbols]

    def __len__(self):
        return len(self.symbols)

    def __repr__(self):
        # return f"[Source({self.symbols})]"
        return self.name


class CodingResults:
    def __init__(self, data, bits):
        """
        :param data: data needed to reconstruct the input
        :param bits: bits needed to represent the input
        """
        self.data = data
        self.bits = bits


class Codec:
    def __init__(self, is_raw=False, build=True):
        self.is_raw = is_raw
        if build:
            self.build()

    def build(self):
        """Run after __init__ if build is True, normally to initialize
        coding structures based on the initialization parameters.
        """
        pass

    def code(self, input, **kwargs):
        """Code an iterable of symbols.

        :return a CodingResults instance
        """
        if self.is_raw:
            return self.code_raw(input=input)
        raise NotImplementedError()

    def decode(self, coded_output, **kwargs):
        """Decode the coded_output of a call to self.code
        :return a list symbols (should be equal to the
          original input for lossless codecs)
        """
        if self.is_raw:
            return self.decode_raw(coded_output=coded_output)
        raise NotImplementedError()

    @property
    def params(self):
        """Get a dictionary with all needed initialization information
        and a '_cls' field with the class whose __init__ is to be invoked.
        """
        d = {"_cls": self.__class__, "is_raw": self.is_raw}
        d.update(self.get_initialization_params())
        return d

    def get_initialization_params(self):
        """:return: a dictionary with the initialization parameters needed to
          produce this codec. This dictionary is passed as argument to self.finish_loading
          after a build=False initialization.
        """
        raise NotImplementedError()

    def code_raw(self, input):
        counter = collections.Counter(input)
        word_length = math.ceil(math.log2(len(counter.items())))
        return {"coded_data": input, "coded_bits": len(input) * word_length}

    def decode_raw(self, coded_output):
        return coded_output["coded_data"]


class Node:
    def __init__(self, symbol, parent=None):
        """
        :param symbol: symbol of the edge connecting with the parent
        :param parent: parent node, or None for the root node
        """
        self.symbol = symbol
        self.parent = parent
        self.symbol_to_node = {}

    def add_child(self, symbol):
        """Add a new child and return the created node
        """
        assert symbol not in self.symbol_to_node, \
            f"{self.__repr__()} already contained a child for symbol {symbol}"
        node = Node(symbol=symbol, parent=self)
        self.symbol_to_node[symbol] = node
        return node

    def show(self, level=0, sep="    "):
        print(f"{self}")
        for symbol, node in self.symbol_to_node.items():
            sys.stdout.write(sep * (level))
            sys.stdout.write("+" + "-" * len(sep))
            node.show(level=level + 1, sep=sep)

    @property
    def word(self):
        nodes = [self]
        while nodes[-1].parent is not None:
            nodes.append(nodes[-1].parent)
        return tuple(reversed([node.symbol for node in nodes if node is not None and node.parent is not None]))

    @property
    def word_probability(self):
        """Probability of a node's word, subtracting children probability
        (Marlin's)
        """
        # Probability of the symbols occurring
        p = 1
        for symbol in self.word:
            p *= symbol.p

        # Probability of a longer child being matched
        p *= (1 - sum(symbol.p for symbol in self.symbol_to_node.keys()))

        return round(p, 12)

    @property
    def raw_word_probability(self):
        """Probability of a node's word, WITHOUT subtracting children probability
        (Yamamotos's)
        """
        # Probability of the symbols occurring
        p = 1
        for symbol in self.word:
            p *= symbol.p
        return round(p, 12)

    @property
    def all_children(self):
        nodes = [self] if self.parent is not None else []
        for c in self.symbol_to_node.values():
            nodes += c.all_children

        return nodes

    def __repr__(self):
        return f"Node(symbol={self.symbol}, parent={self.parent}, #ch={len(self.symbol_to_node)}, word={self.word})"

    def __str__(self):
        return ''.join(str(s) for s in self.word)

    def __gt__(self, other):
        return id(self) > id(other)

    def __lt__(self, other):
        return id(self) < id(other)

    def __hash__(self):
        return id(self)


class Tree(Codec):
    def __init__(self, size: int, source: Source, build=True, **kwargs):
        """Build a code dictionary for the given source

        :param size: number of nodes to be had in the tree's dictionary
        """
        self.source = source
        assert size >= len(self.source.symbols), (size, len(self.source.symbols))
        self.size = size
        self.root = Node(symbol=Symbol(label="root", p=0))

        super().__init__(build=False, **kwargs)

        if build:
            self.build()
            assert len(self.included_nodes) <= self.size
            if len(self.included_nodes) != self.size:
                print(f"WARNING: {self.__class__.__name__} produced {len(self.included_nodes)} "
                      f"<= size {self.size} symbols")

    def deep_copy(self):
        """:return a Tree with new nodes having the same structure and associated symbols
        """
        node_to_new_node = {None: None}
        pending_nodes = [self.root]
        while pending_nodes:
            next_node = pending_nodes.pop()
            new_parent = node_to_new_node[next_node.parent]
            new_node = new_parent.add_child(next_node.symbol) \
                if new_parent is not None else Node(symbol=next_node.symbol, parent=None)
            node_to_new_node[next_node] = new_node
            pending_nodes.extend(next_node.symbol_to_node.values())

        new_tree = Tree(size=self.size, source=self.source, build=False)
        new_tree.root = node_to_new_node[self.root]
        return new_tree, node_to_new_node

    def dump(self, output_path):
        return TreeWrapper(self).dump(output_path)

    def finish_loading(self, loaded_params):
        return self

    def build(self):
        raise NotImplementedError()

    def code_one_word(self, input):
        """:return node, remaining_input,
          where node is the node with the longest word in self.included_nodes that prefixes input
        """
        current_index = 0
        current_node = self.root
        try:
            while True:
                current_symbol = input[current_index]
                current_node = current_node.symbol_to_node[current_symbol]
                current_index = current_index + 1
        except (KeyError, IndexError) as ex:
            assert current_index > 0 and current_node is not self.root, \
                "Non complete dictionary?"
            return current_node, input[current_index:]
        except IndexError as ex:
            raise ex

    def code(self, input):
        """:return coded_output, output_bits
        """
        if self.is_raw:
            return self.code_raw(input)

        emitted_nodes = []
        while len(input) > 0:
            node, new_input = self.code_one_word(input)
            node_word = node.word
            assert len(node_word) > 0
            assert all(n == i for n, i in zip(node_word, input))
            emitted_nodes.append(node)
            input = input[len(node.word):]

        coded_dict = {
            "coded_data": emitted_nodes,
            "coded_bits": len(emitted_nodes) * self.word_length_bits
        }
        return coded_dict

    def decode(self, coded_dict):
        if self.is_raw:
            return self.decode_raw(coded_output=coded_dict)

        return list(itertools.chain(*(node.word for node in coded_dict["coded_data"])))

    @property
    def included_nodes(self):
        return [c for c in self.root.all_children
                if len(c.symbol_to_node) < len(self.source.symbols)]

    @property
    def average_length(self):
        p_sum = 0
        avg_len = 0
        for node in self.included_nodes:
            node_p, word_length = node.word_probability, len(node.word)
            p_sum += node_p
            avg_len += node_p * word_length
        assert round(p_sum, 7) == 1, p_sum
        # return round(avg_len, 12)
        return avg_len

    @property
    def word_length_bits(self):
        return max(1, math.ceil(math.log2(len(self.included_nodes))))

    def get_initialization_params(self):
        return {"size": self.size, "source": self.source}

    @property
    def name(self):
        ignored_params = ["_cls"]
        return f"{self.__class__.__name__}" \
               + ("_" if any(k not in ignored_params for k in self.params.keys()) else "") \
               + "_".join(f"{k}-{v}" if k != "source" else f"{k}-{v.name}"
                          for k, v in sorted(self.params.items())
                          if k not in ignored_params)

    def __repr__(self):
        return f"[{self.__class__.__name__} size={self.size} source={self.source.name}]"


class TrivialTree(Tree):
    def __init__(self, size: int, source: Source, build=True, **kwargs):
        super().__init__(size=size, source=source, build=build, **kwargs)

    def build(self):
        # Basic initialization
        for symbol in self.source.symbols:
            self.root.add_child(symbol)


class MarlinBaseTree(Tree):
    """GetTree function - no Markov optimization
    """

    def __init__(self, size: int, source: Source, build=True, **kwargs):
        super().__init__(size=size, source=source, build=build, **kwargs)

    def node_probability(self, node):
        return node.word_probability

    def build(self):
        # Basic initialization
        p_node_heap = [(-self.node_probability(node), node)
                       for symbol in self.source.symbols
                       for node in [self.root.add_child(symbol=symbol)]]
        heapq.heapify(p_node_heap)

        while len(p_node_heap) < self.size:
            # Add new node
            p, expanded_node = heapq.heappop(p_node_heap)
            next_symbol = self.source.symbols[len(expanded_node.symbol_to_node)]
            new_node = expanded_node.add_child(next_symbol)

            # Update heap
            if len(expanded_node.symbol_to_node) < len(self.source) - 1:
                # Still can be further expanded
                heapq.heappush(p_node_heap,
                               (-self.node_probability(expanded_node),
                                expanded_node))
            elif len(self.source) > 1:
                additional_node = expanded_node.add_child(self.source.symbols[-1])
                heapq.heappush(p_node_heap,
                               (-self.node_probability(additional_node),
                                additional_node))

            heapq.heappush(p_node_heap,
                           (-self.node_probability(new_node),
                            new_node))
            # p_node_heap is incremented in exactly one element after each loop


class MarlinTreeMarkov(MarlinBaseTree):
    """Marlin's code tree, optimized iteratively for stationarity
    """

    def __init__(self, size: int, source: Source, build=True, **kwargs):
        self.state_probabilities = [1] + [0] * (len(source.symbols) - 2)
        # For conditional probability calculation
        self.state_probability_divisors = [1] + [sum(s.p for s in source.symbols[i:])
                                                 for i in range(1, len(source.symbols) - 1)]
        super().__init__(size=size, source=source, build=build, **kwargs)

    def node_conditional_probability(self, node, state_index, state_tree):
        word_probability = node.word_probability
        state_probability = self.state_probabilities[state_index] if self is state_tree else 0
        state_divisor = self.state_probability_divisors[state_index]
        first_symbol_index = self.source.symbols.index(node.word[0])
        if state_probability > 0:
            cp = (state_probability * word_probability / state_divisor) \
                if first_symbol_index >= state_index and state_divisor > 0 else 0
        else:
            cp = 0

        return cp

    def node_probability(self, node):
        return sum(self.node_conditional_probability(node=node, state_index=i, state_tree=self)
                   for i in range(len(self.state_probabilities)))

    def calculate_state_probabilities(self):
        self.state_probabilities = [1] + [0] * (len(self.state_probabilities) - 1)

        # Some sanity checks
        p_sum_by_i = {}
        for i in range(len(self.state_probabilities)):
            p_sum_by_i[i] = 0
            for node in self.included_nodes:
                p_sum_by_i[i] += self.node_conditional_probability(node=node, state_index=i, state_tree=self)
        assert abs(sum(p_sum_by_i.values()) - 1) < 1e-8, sum(p_sum_by_i.values())
        #
        total_s = 0
        for i in range(len(self.state_probabilities)):
            s = 0
            for node in self.included_nodes:
                s += self.node_conditional_probability(node=node, state_index=i, state_tree=self)
            total_s += s
        assert abs(total_s - 1) < 1e-8, math.log10(total_s)

        previous_state_probabilities = self.state_probabilities

        try:
            transition_matrix = np.zeros((len(self.state_probabilities),
                                          len(self.state_probabilities)))
            for i in range(len(self.state_probabilities)):
                for node in self.included_nodes:
                    j = len(node.symbol_to_node)
                    transition_matrix[i, j] += \
                        self.node_conditional_probability(node=node, state_index=i, state_tree=self) / \
                        self.state_probabilities[i] \
                            if self.state_probabilities[i] else 0
                if (transition_matrix[i, :] == 0).all():
                    transition_matrix[i, :] = 1
                    transition_matrix[i, :] /= np.sum(transition_matrix[i, :])

            # assert abs(np.sum(transition_matrix) - len(self.state_probabilities)) < 5e-6, \
            #     (transition_matrix, np.sum(transition_matrix), len(self.state_probabilities))

            iteration_count = 0
            for i in range(100):
                iteration_count += 1
                new_state_probabilities = [0] * len(self.state_probabilities)
                for j in range(len(self.state_probabilities)):
                    for i in range(len(self.state_probabilities)):
                        new_state_probabilities[j] += self.state_probabilities[i] * transition_matrix[i, j]

                delta = sum(abs(old - new) for old, new in zip(
                    self.state_probabilities, new_state_probabilities))
                # assert abs(sum(new_state_probabilities) - 1) < 1e-8, sum(new_state_probabilities)
                tolerance = 5e-7
                if delta < tolerance:
                    break
                self.state_probabilities = new_state_probabilities
            else:
                raise Exception(f"Couldn't converge calculating state probabilities {delta}")

            return self.state_probabilities

        finally:
            self.state_probilities = previous_state_probabilities

    def build(self, update_state_probabilities=True):
        prev_state_probabilities = self.state_probabilities
        super().build()

        previous_words = set()
        for iteration_count in itertools.count(0):
            if update_state_probabilities:
                self.state_probabilities = self.calculate_state_probabilities()
            self.root = Node(symbol=self.root.symbol, parent=None)
            super().build()

            new_words = ",".join(sorted((str(node.word) for node in self.included_nodes)))
            if new_words in previous_words:
                break
            else:
                previous_words.add(new_words)

            tolerance = 1e-10 * ((iteration_count + 1) / 10)
            delta = sum(abs(old - new) for old, new in zip(
                prev_state_probabilities, self.state_probabilities))
            if delta <= tolerance:
                break

            prev_state_probabilities = self.state_probabilities


class YamamotoTree(Tree):
    """[1] YAMAMOTO AND YOKOO:
    AVERAGE-SENSE OPTIMALITY AND COMPETITIVE OPTIMALITY FOR VF CODES
    """

    def __init__(self, size: int, source: Source, build=True,
                 first_allowed_symbol_index=0, **kwargs):
        self.first_allowed_symbol_index = first_allowed_symbol_index
        assert 0 <= self.first_allowed_symbol_index <= len(source.symbols) - 2
        super().__init__(size=size, source=source, build=build, **kwargs)

    def next_child_raw_probability(self, node):
        return node.raw_word_probability * self.source.symbols[
            len(node.symbol_to_node)].p

    def build(self):
        # Basic initialization
        node_list = [self.root.add_child(symbol) for symbol in self.source.symbols[self.first_allowed_symbol_index:]]
        # hat_n : based on raw word probability and maximum degree
        hat_n_heap = [(-node.raw_word_probability, -len(node.symbol_to_node), node)
                      for node in node_list]
        heapq.heapify(hat_n_heap)
        # tilde_n : based on the probability of the next available child
        tilde_n_heap = [(-self.next_child_raw_probability(node), node)
                        for node in node_list]
        heapq.heapify(tilde_n_heap)
        node_count = len(node_list)
        del node_list

        while node_count < self.size:
            _, _, hat_n = hat_n_heap[0]
            expansion_count = max(0, len(self.source.symbols) - len(hat_n.symbol_to_node) - 1)
            assert expansion_count > 0

            S1, S2 = 0, 0  # Gains as defined in [1]

            if node_count + expansion_count <= self.size:
                # S1 - Calculate gain of expanding the most probable node
                S1 = hat_n.raw_word_probability * sum(
                    symbol.p for symbol in self.source.symbols[-expansion_count - 1:])

                # S2 - Calculate gain of expanding the expansion_count best nodes
                # but do it in a copy in case S2 is not accepted
                # s2_root = copy.deepcopy(self.root, memo)
                s2, memo = self.deep_copy()
                s2_root = s2.root
                s2_hat_heap = [(t[0], t[1], memo[t[2]]) for t in hat_n_heap]
                s2_tilde_heap = [(t[0], memo[t[1]]) for t in tilde_n_heap]
            else:
                # Not enough free words to expand - S2 happening for sure
                expansion_count = self.size - node_count
                # no need to deepcopy
                s2_root = self.root
                s2_hat_heap = hat_n_heap
                s2_tilde_heap = tilde_n_heap

            for i in range(expansion_count):
                # Add child node
                ## Remove expanded from queues
                _, tilde_n = heapq.heappop(s2_tilde_heap)
                s2_hat_heap = list(filter(lambda t: t[2] != tilde_n, s2_hat_heap))

                ## Add node to expanded
                new_node = tilde_n.add_child(self.source.symbols[
                                                 len(tilde_n.symbol_to_node)])
                S2 += new_node.raw_word_probability
                # Automatically expand last child if possible
                if len(tilde_n.symbol_to_node) == len(self.source.symbols) - 1:
                    tilde_n = tilde_n.add_child(self.source.symbols[-1])
                    S2 += tilde_n.raw_word_probability

                # Update heaps
                ## Expanded node (or automatic expansion)
                heapq.heappush(s2_hat_heap,
                               (-tilde_n.raw_word_probability, len(tilde_n.symbol_to_node), tilde_n))
                heapq.heappush(s2_tilde_heap,
                               (-self.next_child_raw_probability(tilde_n), tilde_n))
                ## New node
                heapq.heappush(s2_hat_heap,
                               (-new_node.raw_word_probability, len(new_node.symbol_to_node), new_node))
                heapq.heappush(s2_tilde_heap,
                               (-self.next_child_raw_probability(new_node), new_node))

            if S1 >= S2:
                assert hat_n == heapq.heappop(hat_n_heap)[2]
                len_before = len(tilde_n_heap)
                tilde_n_heap = [t for t in tilde_n_heap if t[1] != hat_n]
                assert len(tilde_n_heap) == len_before - 1

                for symbol in self.source.symbols[-expansion_count - 1:]:
                    if len(self.included_nodes) == self.size:
                        break
                    new_node = hat_n.add_child(symbol)
                    heapq.heappush(hat_n_heap,
                                   (-new_node.raw_word_probability, 0, new_node))
                    heapq.heappush(tilde_n_heap,
                                   (-self.next_child_raw_probability(new_node), new_node))
                else:
                    assert len(hat_n.symbol_to_node) == len(self.source.symbols), \
                        (len(hat_n.symbol_to_node), len(self.source.symbols))
            else:
                self.root = s2_root
                hat_n_heap = s2_hat_heap
                tilde_n_heap = s2_tilde_heap

            node_count += expansion_count


class Forest(Codec):

    def __init__(self, tree_sizes, source: Source, build=True, **kwargs):
        """For each value in tree_sizes, a tree is created with at most that size
        """
        if tree_sizes is None and source is None:
            # Empty tree
            return

        self.is_raw = (len(source) == 1)

        assert all(s > 0 for s in tree_sizes)
        self.tree_sizes = list(tree_sizes)
        self.trees = []
        self.current_tree = None
        self.word_to_next_tree = {}
        self.source = source

        super().__init__(build=build)

    def build(self):
        raise NotImplementedError()

    def finish_loading(self, loaded_params):
        """Re-build any needed structures after loading parameters in Forest.load()
        """
        return self

    def code_one_word(self, input):
        node, new_input = self.current_tree.code_one_word(input)
        try:
            self.current_tree = self.word_to_next_tree[node.word]
        except KeyError as ex:
            if len(new_input) == 0:
                self.current_tree = None
            else:
                raise ex
        return node, new_input

    def code(self, input):
        if self.is_raw:
            return self.code_raw(input=input)

        emitted_nodes = []
        root_nodes = set(tree.root for tree in self.trees)
        while len(input) > 0 and self.current_tree not in root_nodes:
            new_node, input = self.code_one_word(input)
            emitted_nodes.append(new_node)

        coded_dict = {
            "coded_data": emitted_nodes,
            "coded_bits": len(emitted_nodes) * self.word_length_bits
        }
        return coded_dict

    def decode(self, coded_output):
        if self.is_raw:
            return self.decode_raw(coded_output=coded_output)
        return list(itertools.chain(*(node.word for node in coded_output["coded_data"])))

    @property
    def included_nodes(self):
        return list(itertools.chain(*(tree.included_nodes for tree in self.trees)))

    def dump(self, output_path):
        assert output_path.split(".")[-1] == "zip"
        output_dir = f"tmp_dump_dir_pid{os.getpid()}"
        os.makedirs(output_dir)
        try:
            tree_node_list = [tree.included_nodes for tree in self.trees]
            tree_word_list = [[node.word for node in tree_nodes] for tree_nodes in tree_node_list]
            flat_word_list = list(itertools.chain(*tree_word_list))

            next_tree_by_word_index = [self.trees[self.trees.index(self.word_to_next_tree[word])]
                                       for word in flat_word_list]

            pickle.dump(self.source, open(os.path.join(output_dir, "source.pickle"), "wb"))
            pickle.dump(tree_word_list, open(os.path.join(output_dir, "tree_word_list.pickle"), "wb"))
            pickle.dump(next_tree_by_word_index, open(os.path.join(output_dir, "next_tree_by_word_index.pickle"), "wb"))
            pickle.dump(self.params, open(os.path.join(output_dir, "params.pickle"), "wb"))
            shutil.make_archive(base_name=output_path.replace(".zip", ""), root_dir=output_dir, format="zip")
        finally:
            shutil.rmtree(output_dir)

    @staticmethod
    def load(input_path):
        input_dir = f"tmp_load_dir_PID{os.getpid()}"
        shutil.unpack_archive(filename=input_path, extract_dir=input_dir, format="zip")
        try:
            source = pickle.load(open(os.path.join(input_dir, "source.pickle"), "rb"))
            tree_word_list = pickle.load(open(os.path.join(input_dir, "tree_word_list.pickle"), "rb"))
            next_tree_by_word_index = pickle.load(open(os.path.join(input_dir, "next_tree_by_word_index.pickle"), "rb"))
            params = pickle.load(open(os.path.join(input_dir, "params.pickle"), "rb"))
            flat_word_list = list(itertools.chain(*tree_word_list))

            cls = params.pop("_cls")
            params["build"] = False
            forest = cls(**params)
            del params["build"]
            params["_cls"] = cls
            # forest.params = params
            forest.source = source
            forest.trees = []
            for tree_words in tree_word_list:
                forest.trees.append(Tree(size=len(tree_words), source=source, build=False))
                forest.trees[-1].root = Node(symbol=Symbol(label="root", p=0))
                for word in tree_words:
                    current_node = forest.trees[-1].root
                    for symbol in word:
                        try:
                            current_node = current_node.symbol_to_node[symbol]
                        except KeyError:
                            current_node = current_node.add_child(symbol)
                stw = sorted(tree_words, key=lambda word: str(word))
                siw_nodes = sorted(forest.trees[-1].included_nodes, key=lambda node: str(node.word))
                siw_words = [node.word for node in siw_nodes]

                assert len(stw) == len(siw_nodes), (len(stw), len(siw_nodes))
                for i, (t, w) in enumerate(zip(stw, siw_words)):
                    assert t == w, (i, t, w)
            forest.tree_sizes = [len(nodes) for nodes in tree_word_list]
            forest.current_tree = forest.trees[0]

            assert len(flat_word_list) == len(next_tree_by_word_index)
            forest.word_to_next_tree = {}
            for word, next_tree in zip(flat_word_list, next_tree_by_word_index):
                forest.word_to_next_tree[word] = next_tree

            # Give a chance to codecs to rebuild any needed internal structures
            returned_forest = forest.finish_loading(loaded_params=params)

            assert abs(sum(forest.source.probabilities) - 1) < 1e-10

            return returned_forest
        finally:
            shutil.rmtree(input_dir)

    @property
    def word_length_bits(self):
        return max(tree.word_length_bits for tree in self.trees)

    @property
    def params(self):
        try:
            return self._params
        except AttributeError:
            return {"tree_sizes": self.tree_sizes,
                    "is_raw": self.is_raw,
                    "source": self.source,
                    "_cls": self.__class__}

    @params.setter
    def params(self, new_params):
        self._params = new_params
        for k, v in new_params.items():
            self.__setattr__(k, v)

    @property
    def name(self):
        ignored_params = ["_cls"]
        param_parts = []
        for param_name, param_value in sorted(self.params.items()):
            if param_name == "source":
                param_parts.append(f"{param_name}-{param_value.name}")
            elif param_name in ignored_params:
                continue
            else:
                param_parts.append(f"{param_name}-{param_value}")
        return f"{self.__class__.__name__}" \
               + ("_" if any(k not in ignored_params for k in self.params.keys()) else "") \
               + "_".join(param_parts)

    def __repr__(self):
        return f"[{self.__class__.__name__} " \
               f"|F|={len(self.trees)} " \
               f"|T|={max(tree.size for tree in self.trees)}" \
               f" source={self.source}" \
               f"]"


class TreeWrapper(Forest):
    def __init__(self, tree, build=True, **kwargs):
        if build:
            self.is_raw = tree.is_raw
            self.source = tree.source
            self.trees = [tree]
            self.tree_sizes = [tree.size]
            self.word_to_next_tree = {node.word: tree for node in tree.included_nodes}

    @property
    def params(self):
        d = super().params
        d["tree"] = self.trees[0]
        return d

    def finish_loading(self, loaded_params):
        tree_params = dict(loaded_params["tree"].params)
        cls = tree_params.pop("_cls")
        tree = cls(build=False, **(tree_params))
        tree.root = self.trees[0].root
        return tree.finish_loading(loaded_params=loaded_params)


class TrivialForest(Forest):
    def build(self):
        for size in self.tree_sizes:
            self.trees.append(TrivialTree(size=size, source=self.source))

        for tree in self.trees:
            for i, node in enumerate(tree.included_nodes):
                self.word_to_next_tree[node.word] = self.trees[i % len(self.trees)]

        self.current_tree = self.trees[0]


class YamamotoForest(Forest):
    """A Yamamoto forest as described in their paper - with exactly |A|-1 trees"""

    def __init__(self, size, source: Source, **kwargs):
        """Create a Yamamoto forest with at most size total nodes
        """
        # |A|-1 states
        tree_count = len(source.symbols) - 1
        tree_sizes = [size] * tree_count
        self.size = size
        super().__init__(tree_sizes=tree_sizes, source=source, **kwargs)

    def build(self):
        for i, size in enumerate(self.tree_sizes):
            if show_progress:
                sys.stdout.write(".")
                sys.stdout.flush()
            self.trees.append(YamamotoTree(size=size, source=self.source, first_allowed_symbol_index=i))
        for node in self.included_nodes:
            self.word_to_next_tree[node.word] = self.trees[0]
        self.current_tree = self.trees[0]

    @property
    def params(self):
        try:
            return self._params
        except AttributeError:
            return {"size": self.size,
                    "source": self.source,
                    "is_raw": self.is_raw,
                    "_cls": self.__class__}

    @params.setter
    def params(self, new_params):
        self._params = new_params
        for k, v in new_params.items():
            self.__setattr__(k, v)


class MarlinForestMarkov(Forest):
    """Proposed Markov forest"""

    def __init__(self, K, O, S, source, symbol_p_threshold=0,
                 original_source=None, build=True, is_raw=False,
                 **kwargs):
        """
        :param K: trees have size 2**K
        :param O: 2**O trees are created
        :param S: shift parameter
        :param source: source model for which the dictionary is to be optimized
        :param symbol_p_threshold: less probably symbols are dropped until the total probability of dropped symbols
          reaches but does not exceed symbol_p_threshold. Dropped symbols are output raw separately. symbol_p_threshold=0
          means that no symbol is coded raw
        """

        self.K = K
        assert 2 ** self.K >= len(source.symbols)
        self.O = O
        assert self.O >= 0
        self.S = S
        assert S >= 0
        assert self.K > self.O  # To meet the required number of transitions between trees

        self.is_raw = is_raw
        self.original_source = Source(symbols=source.symbols) if original_source is None else original_source
        self.symbol_p_threshold = symbol_p_threshold
        assert 0 <= self.symbol_p_threshold <= 1

        full_meta_source, \
        self.metasymbol_to_symbols, \
        self.symbol_to_metasymbol_remainder = \
            Source.get_meta(source=source, symbols_per_meta=2 ** self.S)
        meta_symbols = full_meta_source.symbols

        assert abs(sum(s.p for s in meta_symbols) + - 1) < 1e-10, sum(s.p for s in meta_symbols)

        # Meta-symbols are filtered based on the probability threshold
        selected_symbols = []
        selected_p_sum = 0
        while selected_p_sum < (1 - symbol_p_threshold) \
                and len(selected_symbols) < len(meta_symbols):
            selected_symbols = meta_symbols[:len(selected_symbols) + 1]
            selected_p_sum = sum(ss.p for ss in selected_symbols)
        source = Source([Symbol(label=ss.label, p=ss.p / selected_p_sum)
                         for ss in selected_symbols])

        # selected_symbol_ratio = len(source.symbols) / len(meta_symbols)
        auxiliary_root = Node(symbol=None, parent=None)
        self.symbol_to_auxiliary_node = {
            non_selected_symbol: auxiliary_root.add_child(symbol=non_selected_symbol)
            for non_selected_symbol in meta_symbols[len(selected_symbols):]}
        super().__init__(tree_sizes=[2 ** K] * (2 ** O), source=source, build=False, **kwargs)

        if build:
            self.build()

        assert abs(sum(self.source.probabilities) - 1) < 1e-10

    def finish_loading(self, loaded_params):
        self.is_raw = (len(self.source) == 1)
        assert 0 <= self.symbol_p_threshold <= 1
        assert self.S >= 0

        meta_source, \
        self.metasymbol_to_symbols, \
        self.symbol_to_metasymbol_remainder = \
            Source.get_meta(source=self.original_source,
                            symbols_per_meta=2 ** self.S)
        meta_symbols = meta_source.symbols

        assert len(meta_symbols) == math.ceil(len(self.original_source) / 2 ** self.S)

        selected_symbols = []
        selected_p_sum = 0
        while selected_p_sum < (1 - self.symbol_p_threshold) \
                and len(selected_symbols) < len(meta_symbols):
            selected_symbols = meta_symbols[:len(selected_symbols) + 1]
            selected_p_sum = sum(ss.p for ss in selected_symbols)

        auxiliary_root = Node(symbol=None, parent=None)
        self.symbol_to_auxiliary_node = {
            non_selected_symbol: auxiliary_root.add_child(symbol=non_selected_symbol)
            for non_selected_symbol in meta_symbols[len(selected_symbols):]}

        for symbol in self.original_source.symbols:
            meta_symbol, remainder = self.symbol_to_metasymbol_remainder[symbol]
            recovered_symbol = self.metasymbol_to_symbols[meta_symbol][remainder]
            assert recovered_symbol is symbol

        self.source = meta_source
        self.source.symbols = self.source.symbols[:len(selected_symbols)]
        source_psum = sum(self.source.probabilities)
        for s in self.source.symbols:
            s.p /= source_psum
        #
        return self

    def tree_index_to_tmatrix_index(self, tree, index):
        assert tree in self.trees
        assert 0 <= index < (len(self.source.symbols) - 1), (index, len(self.source.symbols))
        # states_per_tree * tree_index + index_in_tree
        return (len(self.source.symbols) - 1) * self.trees.index(tree) + index

    def tmatrix_index_to_tree_index(self, tmatrix_index):
        tree = self.trees[tmatrix_index // (len(self.source.symbols) - 1)]
        state_index = tmatrix_index % (len(self.source.symbols) - 1)
        return tree, state_index

    def build(self, max_iterations=30):
        # Initial tree building and linking
        for size in self.tree_sizes:
            if show_progress:
                sys.stdout.write(",")
                sys.stdout.flush()
            self.trees.append(MarlinTreeMarkov(size=size if len(self.source) > 1 else 1,
                                               source=self.source))
        if len(self.source) == 1:
            self.is_raw = True
            self.trees = self.trees[:1]
            self.word_to_next_tree = {node.word: self.trees[0] for node in self.trees[0].included_nodes}
            self.current_tree = self.trees[0]
        else:
            self.word_to_next_tree = self.get_word_to_next_tree()
            states_per_tree = len(self.source.symbols) - 1
            state_count = states_per_tree * 2 ** self.O
            state_probabilities = np.zeros(state_count)
            state_probabilities[:] = 1 / state_count
            produced_word_lists = []

            assert len(self.source) > 1
            min_delta = float("inf")
            for iteration_index in range(max_iterations):
                # Transition matrix
                tmatrix = np.zeros((state_count, state_count))
                if show_progress:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                for from_tree in self.trees:
                    for from_index in range(states_per_tree):
                        row = self.tree_index_to_tmatrix_index(tree=from_tree, index=from_index)
                        for node in from_tree.included_nodes:
                            to_tree = self.word_to_next_tree[node.word]
                            to_index = min(state_count - 1, len(node.symbol_to_node))
                            col = self.tree_index_to_tmatrix_index(tree=to_tree, index=to_index)

                            tmatrix[row, col] += from_tree.node_conditional_probability(
                                node=node, state_index=from_index, state_tree=from_tree)
                for row in range(state_count):
                    s = tmatrix[row, :].sum()
                    if s > 0:
                        tmatrix[row, :] /= s
                    else:
                        tmatrix[row, :] = 1 / tmatrix.shape[1]

                # State probability finish_loading
                old_state_probabilities = np.array(state_probabilities)
                while True:
                    for j in range(tmatrix.shape[1]):
                        state_probabilities[j] = 0
                        for transition_p, state_p in zip(tmatrix[:, j], old_state_probabilities):
                            state_probabilities[
                                j] += state_p * transition_p  # P(transition) = sum_state P(transition|state)
                    delta = sum(abs(old - new) for old, new in zip(old_state_probabilities, state_probabilities))
                    min_delta = min(delta, min_delta)
                    if delta < 1e-8:
                        break
                    old_state_probabilities = np.array(state_probabilities)
                mse = 0
                for i, state_p in enumerate(state_probabilities):
                    tree, state_index = self.tmatrix_index_to_tree_index(i)
                    mse += np.abs(np.array(state_p)
                                  - np.array(tree.state_probabilities[state_index])).sum() ** 2
                    tree.state_probabilities[state_index] = state_p
                mse /= state_count

                # Update tree
                all_equal = True
                for tree in self.trees:
                    previous_words = sorted((node.word for node in tree.included_nodes),
                                            key=lambda w: str(w))
                    tree.root = Node(Symbol("root", 0), parent=None)
                    if show_progress:
                        sys.stdout.write("-")
                        sys.stdout.flush()
                    tree.build(update_state_probabilities=False)
                    new_words = sorted((node.word for node in tree.included_nodes),
                                       key=lambda w: str(w))
                    if previous_words != new_words:
                        all_equal = False

                self.word_to_next_tree = self.get_word_to_next_tree()

                new_words = sorted((node.word for node in self.included_nodes),
                                   key=lambda word: str(word))

                tolerance = 1e-6
                if all_equal \
                        or new_words in produced_word_lists \
                        or mse < tolerance:
                    break
                else:
                    produced_word_lists.append(new_words)

        self.current_tree = self.trees[0]

    def get_word_to_next_tree(self):
        """Implementation of the DefineTransitions routine described in the paper.
        Determines which words produce transitions to what tree.
        """
        word_to_next_tree = {}
        for tree in self.trees:
            nodes_by_metastate_index = {i: [] for i in range(max(1, len(self.source.symbols) - 1))}
            for node in tree.included_nodes:
                nodes_by_metastate_index[max(0, min(len(self.source.symbols) - 2, len(node.symbol_to_node)))].append(
                    node)
            nodes_by_metastate_index = {i: sorted(nodes, key=lambda node: -tree.node_probability(node))
                                        for i, nodes in nodes_by_metastate_index.items()}

            sorted_nodes = list(
                itertools.chain(*(nodes_by_metastate_index[i] for i in range(len(nodes_by_metastate_index)))))
            assert len(self.trees) == 2 ** self.O
            for i, tree in enumerate(self.trees):
                assert len(tree.included_nodes) == 2 ** self.K
                for node in sorted_nodes[i * 2 ** (self.K - self.O):(i + 1) * 2 ** (self.K - self.O)]:
                    word_to_next_tree[node.word] = tree

        for auxiliary_node in self.symbol_to_auxiliary_node.values():
            word_to_next_tree[auxiliary_node.word] = self.trees[0]

        return word_to_next_tree

    def code(self, input):
        if self.is_raw:
            return self.code_raw(input=input)

        if len(self.source) > 1:
            input, remainders = zip(*(self.symbol_to_metasymbol_remainder[symbol] for symbol in input))

            excluded_index_symbol = [(index, symbol) for index, symbol in enumerate(input)
                                     if symbol not in self.source.symbols]

            excluded_indices = set(index for index, symbol in excluded_index_symbol)

            filtered_input = [symbol for i, symbol in enumerate(input)
                              if i not in excluded_indices]

            coded_dict = super().code(filtered_input)

            coded_nodes = coded_dict["coded_data"]

            proper_word_length = len(coded_nodes) * self.word_length_bits
            excluded_word_length = len(excluded_index_symbol) \
                                   * (self.word_length_bits + math.ceil(math.log2(len(input))))
            raw_data_length = len(remainders) * self.S

            coded_dict = {
                "coded_data": (coded_nodes, excluded_index_symbol, remainders),
                "coded_bits": proper_word_length + excluded_word_length + raw_data_length
            }
        else:
            assert self.is_raw
            coded_dict = {
                "coded_data": ([self.trees[0].root.symbol_to_node[self.source.symbols[0]]] * len(input),
                               [],
                               [0] * len(input),),
                "coded_bits": len(input) * math.ceil(math.log2(len(self.original_source)))
            }
        return coded_dict

    def decode(self, coded_dict):
        if self.is_raw:
            return self.decode_raw(coded_output=coded_dict)

        coded_nodes, excluded_index_symbol, remainders = coded_dict["coded_data"]
        decoded_symbols = list(itertools.chain(*(node.word for node in coded_nodes)))

        for index, symbol in excluded_index_symbol:
            decoded_symbols.insert(index, symbol)

        assert len(decoded_symbols) >= len(remainders), (len(decoded_symbols), len(remainders))
        decoded_symbols = [self.metasymbol_to_symbols[metasymbol][remainder]
                           for metasymbol, remainder in zip(decoded_symbols, remainders)]

        return decoded_symbols

    @property
    def params(self):
        try:
            return self._params
        except AttributeError:
            return {"K": self.K,
                    "O": self.O,
                    "S": self.S,
                    "symbol_p_threshold": self.symbol_p_threshold,
                    "source": self.source,
                    # "metasymbol_to_symbols": self.metasymbol_to_symbols,
                    "original_source": self.original_source,
                    # "symbol_to_auxiliary_node": self.symbol_to_auxiliary_node,
                    # "symbol_to_metasymbol_remainder": self.symbol_to_metasymbol_remainder,
                    "is_raw": self.is_raw,
                    "_cls": self.__class__}

    @params.setter
    def params(self, new_params):
        for k, v in new_params.items():
            self.__setattr__(k, v)

    @property
    def name(self):
        param_parts = []
        for param_name, param_value in sorted(self.params.items()):
            if param_name in ["source", "original_source"]:
                param_parts.append(f"{param_name}-{param_value.name}")
            elif param_name in ["metasymbol_to_symbols",
                                "_cls",
                                "symbol_to_auxiliary_node",
                                "symbol_to_metasymbol_remainder"]:
                continue
            else:
                param_parts.append(f"{param_name}-{param_value}")
        return f"{self.__class__.__name__}" \
               + ("_" if self.params else "") \
               + "_".join(param_parts)

    def __str__(self):
        return f"[MarlinForestMarkov(K={self.K}, O={self.O}, S={self.S}, source={self.source}, symbol_p_trheshold={self.symbol_p_threshold}]"


def entropy(distribution):
    assert round(sum(distribution), 8) == 1
    return sum(-p * math.log(p, 2) for p in distribution if p > 0)


def get_confidence_interval(data, confidence=0.95):
    """USe a ttest to obtain a confidence interval
    :return: (low_end, mean, high_end) for p = 1-conficence
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m, m + h


def compare_basic_trees():
    symbol_counts = [2 ** 8]
    # K_values = list(range(4, 11))
    K_values = [8, 12]
    cls_values = [MarlinBaseTree, YamamotoTree]
    alpha_values = list(np.linspace(0.01, 1, 5)) + list(np.linspace(1, 5, 5))[1:]

    output_path = "comparison_results.csv"
    df = pd.DataFrame(columns=["_cls", "alpha", "K", "avg_len",
                               "time_seconds", "entropy", "symbol_count"])

    def params_to_loc(symbol_count, cls, alpha, K):
        return f"{cls.__name__}_a{alpha:.7f}_K{K}_sc{symbol_count}"

    for symbol_count in symbol_counts:
        for cls in cls_values:
            for alpha in alpha_values:
                for K in K_values:
                    size = 2 ** K
                    if size < symbol_count:
                        continue
                    probabilities = [2 ** (-alpha * i) for i in range(1, symbol_count + 1)]
                    probabilities = [p / sum(probabilities) for p in probabilities]
                    source = Source(symbols=[Symbol(label=chr(ord("A") + i), p=p)
                                             for i, p in enumerate(probabilities)])
                    time_before = time.time()
                    d = cls(size=size, source=source)
                    df.loc[params_to_loc(symbol_count=symbol_count, _cls=cls, alpha=alpha, K=K)] = pd.Series({
                        "_cls": cls.__name__,
                        "alpha": alpha,
                        "K": K,
                        "avg_len": d.average_length,
                        "time_seconds": time.time() - time_before,
                        "entropy": entropy(probabilities),
                        "symbol_count": symbol_count,
                    })

    target_columns = ["avg_len", "time_seconds"]
    anchor_cls = YamamotoTree
    df["anchor"] = anchor_cls.__name__
    for target_column in target_columns:
        df[f"{target_column}_diff"] = float("nan")
        df[f"{target_column}_ratio"] = float("nan")

    for (alpha, K, symbol_count), group in df.groupby(by=["alpha", "K", "symbol_count"]):
        for target_column in target_columns:
            value_by_cls = {cls.__name__: group.loc[
                params_to_loc(symbol_count=symbol_count, _cls=cls, alpha=alpha, K=K)][target_column]
                            for cls in cls_values}
            for cls in cls_values:
                index = params_to_loc(symbol_count=symbol_count, _cls=cls, alpha=alpha, K=K)
                df.at[index, [f"{target_column}_diff", f"{target_column}_ratio"]] = \
                    value_by_cls[cls.__name__] - value_by_cls[anchor_cls.__name__], \
                    value_by_cls[cls.__name__] / value_by_cls[anchor_cls.__name__]
    df.to_csv(output_path)

    original_stdout = sys.stdout
    sys.stdout = open("confidence_interval_analys.md", "w")
    for p in [0.05, 0.01, 0.001]:
        print()
        print()
        print("# p = {}".format(p))
        for cls in cls_values:
            df.loc[f"mean_{cls.__name__}"] = df[df["_cls"] == cls.__name__].mean()

            if cls is not anchor_cls:
                print(f"## cls = {cls.__name__}")
                for target_column in target_columns:
                    print("### target_column = {} - {}".format(target_column,
                                                               df.loc[f"mean_{cls.__name__}"][target_column]))
                    ci_ratio = get_confidence_interval(df[df["_cls"] == cls.__name__][f"{target_column}_ratio"],
                                                       confidence=(1 - p))
                    ci_diff = get_confidence_interval(df[df["_cls"] == cls.__name__][f"{target_column}_diff"],
                                                      confidence=(1 - p))
                    pm_ratio = (ci_ratio[2] - ci_ratio[0]) / 2
                    pm_diff = (ci_diff[2] - ci_diff[0]) / 2
                    print("Ratio:")
                    print()
                    print(f"- avg = {df.loc[f'mean_{cls.__name__}'][f'{target_column}_ratio']}")
                    print(f"- (1 - avg) = {1 - df.loc[f'mean_{cls.__name__}'][f'{target_column}_ratio']}")
                    print(f"- pm = {pm_ratio}")
                    print()
                    print("Diff:")
                    print()
                    print(f"- avg = {df.loc[f'mean_{cls.__name__}'][f'{target_column}_diff']}")
                    print(f"- pm = {pm_diff}")
                    print()
    sys.stdout.close()
    sys.stdout = original_stdout


# --- Simulation running

def codec_dict_source_to_path(codec_params_dict, source):
    codec_params_dict = dict(codec_params_dict)
    cls_name = codec_params_dict.pop("_cls").__name__
    codec_params_dict["source"] = source.name

    path = os.path.join(
        prebuilt_dir,
        f"{cls_name}_" +
        "_".join(f"{k}-{codec_params_dict[k]}" for k in sorted(codec_params_dict.keys()))
        + ".zip")

    return path


def evaluate_codec(codec_params_dict, source, input):
    print(f"\n\n## Evaluating {codec_params_dict} for len(input)={len(input)}...")

    codec_dump_path = codec_dict_source_to_path(codec_params_dict=codec_params_dict, source=source)
    codec_is_loaded = False
    if os.path.isfile(codec_dump_path):
        print(f"\t- loading prebuilt {codec_dump_path}...")
        codec = Forest.load(codec_dump_path)
        codec_is_loaded = True
    else:
        print("\t- initializing codec...")
        codec_params_dict = dict(codec_params_dict)
        cls = codec_params_dict.pop("_cls")
        codec_params_dict["source"] = source
        codec = cls(**codec_params_dict)
    print("\t- codec.name = {}".format(codec.name))

    print("\t- coding input...")
    coded_dict = codec.code(input)
    print("\t- decoding and comparing...")
    decoded_data = codec.decode(coded_dict)[:len(input)]
    decoded_symbols = decoded_data[:len(input)]

    assert np.array(input).tolist() == decoded_symbols, \
        ("Not lossless", codec.name, input[:10], decoded_symbols[:10])
    bps = coded_dict["coded_bits"] / len(input)
    print(f"\t- coded rate: {bps} bps")

    if not codec_is_loaded:
        print(f"\t- dumping prebuilt codec {codec.name} to {codec_dump_path}")
        codec.dump(codec_dump_path)
        delta = 1
        try:
            print("\t- verifying dumped codec...")
            loaded_codec = Forest.load(codec_dump_path)
            loaded_codec_dict = loaded_codec.code(input)
            print(f"\t  loaded_codec = {loaded_codec.name}")
            delta = loaded_codec_dict["coded_bits"] - coded_dict["coded_bits"]
            assert delta == 0, f"Loaded dict does not code equally. Delta = {delta} coded bits"
        finally:
            if delta != 0:
                os.remove(codec_dump_path)

    print(f"\t- codec is --[lossless]--> {bps:.5f}bps")

    return dict(bps=bps, source=source, codec=codec, input=input)


def plot_input_samples(input_by_source):
    for source, input in input_by_source.items():
        plot_path = os.path.join(input_plot_dir, f"input_{source.name}_seed{seed}.pdf")
        if os.path.exists(plot_path):
            continue
        plt.figure()
        plt.hist([source.symbols.index(s) for s in input],
                 bins=len(source), density=True, range=(0, len(source)), rwidth=1,
                 label="sample", edgecolor="black")
        plt.xlim(0, len(source) - 1)
        # x_values, y_values = zip(*((i, symbol.p) for i, symbol in enumerate(source.symbols)))
        # plt.plot(x_values, y_values, label="expected", alpha=0.7)
        plt.xlabel("Input symbol")
        plt.ylabel("Probability")
        plt.legend(loc="best")
        plt.title(f"Alphabet: {len(source)} symbols\n"
                  f"Source: {source.affix if source.affix else 'custom'}, entropy: {source.entropy:.4f} bps\n"
                  f"Sample length: {len(input)} (seed: {hex(seed)})")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


def save_raw_inputs(input_by_source):
    for source, input in input_by_source.items():
        output_path = os.path.join(
            synthetic_inputs_dir,
            f"input_source-{source.name}_len-{len(input)}_seed-0x{hex(seed)}.raw")
        if not os.path.exists(output_path):
            print(f"\t- saving {output_path}")
            input_indices = np.array([source.symbols.index(symbol) for symbol in input])
            assert input_indices.min() >= 0
            assert input_indices.max() <= 255
            input_indices = input_indices.astype("u1")
            input_indices.tofile(output_path)


def generate_efficiency_results():
    """Compute efficiency results and produce a pandas DataFrame with the aquired data
    """

    print("# Generating sources...")
    sources = [Source.get_laplacian(symbol_count=symbol_count, max_entropy_fraction=e)
               for e in reversed(np.linspace(1e-6, 1 / min_max_count_tested_e_fractions[2], 4))] + \
              [Source.get_laplacian(symbol_count=symbol_count, max_entropy_fraction=f)
               for f in np.linspace(*min_max_count_tested_e_fractions)]

    print("# Generating inputs...")
    input_by_source = {source: source.generate_symbols(test_input_length) for source in sources}
    print("# Saving inputs...")
    save_raw_inputs(input_by_source=input_by_source)
    print("# Representing input...")
    plot_input_samples(input_by_source=input_by_source)

    test_tasks = []
    # test_tasks.append(dict(_cls=MarlinForestMarkov, K=8, O=2, S=2, symbol_p_threshold=0))
    # test_tasks.append(dict(_cls=MarlinForestMarkov, K=6, O=0, S=0, symbol_p_threshold=0))
    # test_tasks.append(dict(_cls=MarlinForestMarkov, K=6, O=0, S=0, symbol_p_threshold=1e-6))
    # test_tasks.append(dict(_cls=TrivialTree, size=2**7))
    # test_tasks.append(dict(_cls=TrivialTree, size=2 ** 6, is_raw=True))

    comparison_tasks = []
    min_size_exponent = symbol_count_exponent + 1
    max_size_exponent = symbol_count_exponent + 5
    # # SOTA
    # comparison_tasks += [dict(_cls=YamamotoTree, size=2 ** e) for e in
    #                      reversed(range(min_size_exponent, max_size_exponent + 1))]
    # comparison_tasks += [dict(_cls=YamamotoForest, size=2 ** e) for e in
    #                      reversed(range(min_size_exponent, max_size_exponent + 1))]
    # # Basic Marlin analysis
    # comparison_tasks += [dict(_cls=MarlinBaseTree, size=2 ** e) for e in
    #                      reversed(range(min_size_exponent, max_size_exponent + 1))]
    # # Marlin parameter analysis
    # for K in reversed(range(symbol_count_exponent + 1, 9)):
    #     if minimal_run and (K != 6):
    #         continue
    #     for O in reversed(range(min(symbol_count_exponent + 3, K))):
    #         if minimal_run and (O not in [0, 2]):
    #             continue
    #         for S in reversed(range(symbol_count_exponent)):
    #             if minimal_run and (S not in [0, 2]):
    #                 continue
    #             theta_values = [0] + [10 ** (-x) for x in reversed(list(range(2, 7)))]
    #             for i, theta in enumerate(theta_values):
    #                 comparison_tasks.append(
    #                     dict(_cls=MarlinForestMarkov, K=K, O=O, S=S, symbol_p_threshold=theta))

    ## Fig 1 - Tree codec comparison
    if generate_fig1:
        for K in [8]:
            comparison_tasks.append(dict(_cls=MarlinBaseTree, size=2 ** K))
            comparison_tasks.append(dict(_cls=YamamotoTree, size=2 ** K))
            comparison_tasks.append(dict(_cls=MarlinForestMarkov, K=K, O=0, S=0, symbol_p_threshold=0))

    ## Fig 2 - Forest codec comparison
    if generate_fig2:
        K = 8
        for O in range(symbol_count_exponent):
            comparison_tasks.append(dict(_cls=MarlinForestMarkov,
                                         K=K, O=O, S=0, symbol_p_threshold=0))
        comparison_tasks.append(dict(_cls=YamamotoForest, size=2 ** K))

    ## Fig 3 - Shift and symbol_p_threshold
    if generate_fig3:
        K = 8
        O = 2
        for S in range(symbol_count_exponent):
            comparison_tasks.append(dict(_cls=MarlinForestMarkov,
                                         K=K, O=O, S=S, symbol_p_threshold=0))
            comparison_tasks.append(dict(_cls=MarlinForestMarkov,
                                         K=K, O=O, S=S, symbol_p_threshold=1e-6))
        # comparison_tasks.append(dict(_cls=YamamotoForest, size=2**K))

    print("# Defining tasks...")
    codec_dicts = test_tasks if test_tasks else comparison_tasks
    # codec_dicts = test_tasks + comparison_tasks
    # codec_dicts = comparison_tasks
    for d in codec_dicts:
        task_string = f"{d['_cls'].__name__}(**"
        task_string += str({k: v for k, v in d.items() if k != '_cls'}) + ")"
        print(f"\t- {task_string}")

    kwargs_list = [dict(codec_params_dict=codec_dict,
                        source=source,
                        input=input_by_source[source])
                   for codec_dict in codec_dicts
                   for source in sources]

    print("# Running tasks...")
    invocation_list = [(evaluate_codec, kwargs) for kwargs in kwargs_list]

    df = pd.DataFrame(columns=efficiency_common_columns)

    def append_result(function, kwargs, result):
        source = result["source"]
        codec = result["codec"]
        input = result["input"]
        result["codec"] = codec.name
        result["source"] = result["source"].name

        data_dict = {
            "cls": codec.__class__.__name__,
            "codec": codec.name,
            "codec_path": os.path.basename(codec_dict_source_to_path(
                codec_params_dict=codec.params, source=codec.source)),
            "source": source.name,
            "source_len": len(source),
            "source_entropy": source.entropy,
            "source_affix": source.affix,
            "input_len": len(input),
            "input_entropy": input_entropy(input),
            "bps": result["bps"]
        }
        df.loc[f"{codec.name}_ON_{source.name}"] = pd.Series(data_dict)
        for k, v in codec.params.items():
            try:
                if k in ["source", "original_source"]:
                    v = v.name
                elif k in ["_cls", "is_raw_coder"]:
                    continue
                loc = f"{codec.name}_ON_{source.name}"
                try:
                    df.at[loc, k] = v
                except ValueError:
                    df[k] = df[k].astype(str)
                    df.at[loc, k] = str(v)
            except RecursionError as ex:
                print(f"Error storing param {k}={v}")
                pass

    invoker = pinvoker.ParallelInvoker(
        invocation_list=invocation_list, process_count=process_limit)
    results = invoker.run(
        on_result_func=append_result,
        exit_on_error=exit_on_error,
        verbose=multiprocess_verbose)

    df.to_csv(efficiency_results_path)

    print(f"Obtained {len(results)} results!")
    if len(results) < len(invocation_list):
        print("(less than expected - see messages above)")
        return None
    else:
        return df


# --- Plotting


def curate_data(df):
    """Add dependent columns (i.e., ratios)
    and sanitize types
    """
    df["input_eff_ratio"] = pd.DataFrame(df["input_entropy"] / df["bps"])
    df = df[df["source_entropy"] > 0]
    # df.loc[df["bps"] == 0, "input_eff_ratio"] = 1 # entropy sources 0 sometimes coded for free
    df["source_eff_ratio"] = pd.DataFrame(df["source_entropy"] / df["bps"])

    return df


def plot_efficiency_results(
        full_df,
        x_column_name="input_entropy",
        y_column_name="input_eff_ratio",
        pretty_label_dict={
            'symbol_p_threshold': '$\Theta$',
            'input_eff_ratio': r'Compression efficiency $\eta_\aleph$',
            'input_entropy': r'Source entropy $H(\aleph)$ (bps)',
        }):
    # Filter out raw coders - uninteresting
    full_df["is_raw"] = full_df["is_raw"].fillna(False)
    # df = df[df["is_raw"] == False]

    for source_len, df in full_df.groupby(by="source_len"):
        # Group data by cls then by parameters
        param_columns = [v for v in df.columns.values[len(efficiency_common_columns):].tolist()
                         if v not in efficiency_common_columns
                         and "source" not in v]
        if "cls" not in df.columns:
            print("[watch] df.columns = {}".format(df.columns))

        data_by_cls_sourcelen_label_pvdict = {}
        for cls_name, group_df in df.groupby(by="cls"):
            print("========")
            print(f"{cls_name}")
            print("========")

            subgroup_params = [param for param in param_columns
                               if group_df[param].notnull().any()
                               and param != y_column_name]

            group_df = group_df.sort_values(by=x_column_name)

            int_params = ["size", "K", "O", "S"]

            subgroup_added = False
            for l, sg_df in group_df.groupby(by=list(subgroup_params)):
                # Build subgroup label
                param_value_dict = {}
                try:
                    for i, (param, value) in enumerate(zip(subgroup_params, l)):
                        if param in int_params:
                            value = int(value)
                        param_value_dict[param] = value
                except TypeError:
                    assert len(subgroup_params) <= 1
                    l = l if subgroup_params[0] not in int_params else int(l)
                    param_value_dict[subgroup_params[0]] = l
                label = pretty_label_dict.get(cls_name, cls_name)
                if param_value_dict:
                    label += " :: "
                    label += " - ".join(f"{pretty_label_dict.get(param, param)}:"
                                        f"{pretty_label_dict.get(value, value)}"
                                        for param, value in param_value_dict.items())

                data_by_cls_sourcelen_label_pvdict[(
                    cls_name, source_len, label, frozenset(param_value_dict.items()))] = \
                    LineData(label=label,
                             x_values=sg_df[x_column_name].values,
                             y_values=sg_df[y_column_name].values,
                             x_label=pretty_label_dict.get(x_column_name, x_column_name),
                             y_label=pretty_label_dict.get(y_column_name, y_column_name))
                subgroup_added = True
            if not subgroup_added:
                data_by_cls_sourcelen_label_pvdict[(
                    cls_name, source_len, label, {})] = LineData(
                    label=cls_name,
                    x_values=group_df[x_column_name].values,
                    y_values=group_df[y_column_name].values,
                    x_label=pretty_label_dict.get(x_column_name, x_column_name),
                    y_label=pretty_label_dict.get(y_column_name, y_column_name))

        # Paper comparisons
        if generate_fig1:
            plt.rcParams.update({'font.size': 12})
            ## Base - Markov - Yamamoto
            max_K = 8
            size = 2 ** max_K
            marlin_basetree_data = find_by_cls_params(
                cls=MarlinBaseTree, data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
                params={"size": 2 ** max_K})
            marlin_basetree_data.label = f"Basic Tree (Algorithm 3): $| \mathcal{{T}}\ | = {size:.0f}$"
            marlin_markovtree_data = filter_marlinforest_selection(
                data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
                K=max_K, O=0, S=0, symbol_p_threshold=0)

            assert len(marlin_markovtree_data) >= 1
            marlin_markovtree_data = marlin_markovtree_data[0]
            marlin_markovtree_data.extra_kwargs.update(color="red")
            marlin_markovtree_data.label = f"Stochastically optimized Tree (Algorithm 4): $| \mathcal{{T}}\ | = {size:.0f}$"
            yamamoto_tree_data = find_by_cls_params(
                cls=YamamotoTree, data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
                params={"size": size})
            yamamoto_tree_data.label = f"Yamamoto and Yokoo's Tree: $|\mathcal{{T}}\ | = {size:.0f}$"
            yamamoto_tree_data.extra_kwargs.update(linestyle="--", color="magenta")

            marlin_basetree_data.extra_kwargs.update(dict(linestyle="--"))
            marlin_markovtree_data.extra_kwargs.update(dict(linestyle="--"))


            plt.figure()
            marlin_basetree_data.render()
            marlin_markovtree_data.render()
            yamamoto_tree_data.render()
            plt.xlim(0, symbol_count_exponent)
            plt.ylim(min_shown_efficiency, 1)
            plt.savefig(os.path.join(efficiency_plot_dir, f"markov_improvements_{source_len}symbols.pdf"),
                        bbox_inches="tight")
            plt.close()
            # Same but with diff
            plt.figure()
            plt.hlines(0, 0, symbol_count_exponent, alpha=0.25)
            marlin_basetree_data.diff(yamamoto_tree_data).render()
            basetree_avg_diff = np.mean(marlin_basetree_data.diff(yamamoto_tree_data).y_values)
            marlin_markovtree_data.diff(yamamoto_tree_data).render()
            markovtree_avg_diff = np.mean(marlin_markovtree_data.diff(yamamoto_tree_data).y_values)
            print(f"Base tree - YYtree: {marlin_basetree_data.y_label}: avg diff = {basetree_avg_diff}")
            print(f"Markov tree - YYtree: {marlin_basetree_data.y_label}: avg diff = {markovtree_avg_diff}")
            plt.xlim(0, symbol_count_exponent)
            # plt.ylim(min_shown_efficiency, 1)
            plt.legend()
            plt.savefig(os.path.join(efficiency_plot_dir, f"diff_markov_improvements_{source_len}symbols.pdf"),
                        bbox_inches="tight")
            plt.close()

        if generate_fig2:
            plt.rcParams.update({'font.size': 12})
            max_K = 8
            size = 2 ** max_K
            O_values = range(symbol_count_exponent)
            plt.figure()

            yamamoto_forest_data = find_by_cls_params(
                cls=YamamotoForest,
                data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
                params={"size": size})
            yamamoto_forest_data.label = f"Yamamoto and Yokoo's Forest: $|\mathcal{{F}}\ | = {2 ** symbol_count_exponent - 1}$,  $|\mathcal{{T}}\ | = {size:.0f}$"
            yamamoto_forest_data.alpha = 0.5
            yamamoto_forest_data.extra_kwargs.update(dict(linestyle="-", color="magenta"))

            for O in O_values:
                data = find_by_cls_params(
                    cls=MarlinForestMarkov,
                    data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
                    params={"K": max_K, "O": O})
                data.label = \
                    f"Algorithm 6 Forest: " \
                    f"$| \mathcal{{F}}\ | = 2^\Omega = {2 ** O}$, $| \mathcal{{T}}\ | = 2^K = {2 ** max_K}$"
                data.alpha = 0.5
                if O is O_values[0]:
                    data.extra_kwargs = {"linestyle": "--", "color":"red"}
                elif O is O_values[2]:
                    data.extra_kwargs = {"linestyle": "-", "color": "green"}

                data.render()
                print(f"{data.y_label}: K={max_K} O={O} {np.mean(data.diff(yamamoto_forest_data).y_values)}")

            yamamoto_forest_data.render()
            plt.xlim(0, symbol_count_exponent)
            plt.ylim(min_shown_efficiency, 1)
            plt.savefig(
                os.path.join(efficiency_plot_dir,
                             f"forest_improvements_{source_len}symbols.pdf"), bbox_inches="tight")
            plt.close()

        ## Fig 3 Shift
        if generate_fig3:
            plt.rcParams.update({'font.size': 12})
            K = 8
            O = 2
            size = 2 ** 8
            for S in [0, 1, 3]:
                for symbol_p_threshold in [0]:
                    data = find_by_cls_params(
                        data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
                        cls=MarlinForestMarkov,
                        params={"K": K, "O": O, "S": S, "symbol_p_threshold": symbol_p_threshold})
                    data.label = \
                        f"Algorithm 6 Forest: " \
                        f"$| \mathcal{{F}}\ | = 2^\Omega = {2 ** O}$, " \
                        f"$| \mathcal{{T}}\ | = 2^K = {2 ** K}$, " \
                        f"S = {S} " \
                        ""
                    # f":: $\Theta = {symbol_p_threshold}$" \
                    data.extra_kwargs["linestyle"] = ":" if symbol_p_threshold != 0 else "-"
                    if S == 0:
                        data.extra_kwargs["color"] = "green"
                    elif S == 1:
                        data.extra_kwargs["color"] = "green"
                        data.extra_kwargs["linestyle"] = "-."
                    elif S == 3:
                        data.extra_kwargs["color"] = "green"
                        data.extra_kwargs["linestyle"] = ":"
                    data.alpha = 0.5
                    data.render()

            yamamoto_forest_data = find_by_cls_params(
                cls=YamamotoForest,
                data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
                params={"size": size})
            yamamoto_forest_data.label = f"Yamamoto and Yokoo's Forest: $|\mathcal{{F}}\ | " \
                                         f"= {2 ** symbol_count_exponent - 1}$, " \
                                         f"$|\mathcal{{T}}\ | = {size:.0f}$"
            yamamoto_forest_data.alpha = 0.5
            yamamoto_forest_data.extra_kwargs.update(linestyle="-", color="magenta")
            yamamoto_forest_data.render()
            plt.xlim(0, symbol_count_exponent)
            plt.ylim(min_shown_efficiency, 1)
            plt.savefig(
                os.path.join(efficiency_plot_dir,
                             f"shift_improvements_{source_len}symbols.pdf"), bbox_inches="tight")
            plt.close()


def filter_marlinforest_selection(data_by_cls_sourcelen_label_pvdict, K, O, S, symbol_p_threshold):
    """Return a list of PlottableData instances for the marlin
    results with the given parameters. To accept any value for a
    parameter, set the argument to None.

    :return: list of PlottableData instances
    """
    K_values = np.array(K)
    O_values = np.array(O)
    S_values = np.array(S)
    theta_values = np.array(symbol_p_threshold)

    selected_data = []
    for ((cls_name, source_len, label, pvdict), data) \
            in data_by_cls_sourcelen_label_pvdict.items():
        pvdict = dict(pvdict)
        if cls_name != MarlinForestMarkov.__name__:
            continue
        if (K is None or pvdict["K"] in K_values) \
                and (O is None or pvdict["O"] in O_values) \
                and (S is None or pvdict["S"] in S_values) \
                and (symbol_p_threshold is None or pvdict["symbol_p_threshold"] in theta_values):
            selected_data.append(data)
    return selected_data


def find_by_cls_params(data_by_cls_sourcelen_label_pvdict, cls, params={}):
    for k, data in data_by_cls_sourcelen_label_pvdict.items():
        cls_name, source_len, label, pvdict = k
        pvdict = dict(pvdict)
        if cls_name == cls.__name__:
            for param, value in params.items():
                try:
                    if pvdict[param] != value:
                        break
                except KeyError:
                    break
            else:
                return data
    raise ValueError(f"Cannot find {cls.__name__} with {params}")


def filter_yamamotoforest_selection(data_by_cls_sourcelen_label_pvdict, size):
    return find_by_cls_params(cls=YamamotoForest,
                              data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
                              params={"size": size})


def filter_marlin_basetree(data_by_cls_sourcelen_label_pvdict, size):
    return find_by_cls_params(
        cls=MarlinBaseTree,
        data_by_cls_sourcelen_label_pvdict=data_by_cls_sourcelen_label_pvdict,
        params={"size": size})


if __name__ == '__main__':
    if os.path.exists(efficiency_results_path) and not overwrite_efficiency_results:
        print(f"Re-using result csv {efficiency_results_path}")
        df = pd.read_csv(efficiency_results_path)
    else:
        df = generate_efficiency_results()

    if df is None:
        print("No data to render. Aborting")
        sys.exit(1)

    df = curate_data(df)
    plot_efficiency_results(df)
