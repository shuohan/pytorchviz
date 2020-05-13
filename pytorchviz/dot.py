import torch
from collections import namedtuple
from graphviz import Digraph


NODE_ATTR = dict(style='filled', shape='box', align='left', fontsize='12',
                 ranksep='0.1', height='0.2')
GRAPH_ATTR = dict(size='12,12')
INPUT_COLOR = 'orange'
OUTPUT_COLOR = 'darkolivegreen1'
OP_COLOR = 'antiquewhite'
PARAM_COLOR = 'lightblue'


def make_dot(x, model):
    """Produces Graphviz representation of PyTorch autograd graph.

    Notes:
        * Blue nodes are the Variables that require grad.
        * Orange nodes are input tensors.
        * Green nodes are output tensors.
        * Darkpink nodes are backward operations.

    Args:
        x (torch.Tensor or tuple[torch.Tensor]): The input tensor(s).
        model (torch.nn.Module): The model to visualize.

    Returns:
        graphviz.Digraph: The dot representation.

    """
    return _MakeDot(x, model).make()


class _MakeDot:
    """Produces Graphviz representation of PyTorch autograd graph.

    Attributes:
        x (torch.Tensor or tuple[torch.Tensor]): The input tensor(s).
        y (torch.Tensor or tuple[torch.Tensor]): The output tensor(s).
        model (torch.nn.Module): The model to visualize.

    """
    def __init__(self, x, model):
        self.x = self._convert_input(x)
        self.model = model
        self.y = self._calc_output(self.x, self.model)

        self._visited = set()
        self._get_param_name = _GetParamName(self.model)
        self._get_input_name = _GetInputName(self.x)
        self._get_output_name = _GetOutputName(self.y)
        self._dot = Digraph(node_attr=NODE_ATTR,
                            graph_attr=GRAPH_ATTR)

    def make(self):
        """Produces dot representation of the graph.

        Returns:
            graphviz.Digraph: The dot representation.

        """
        self._make_output_nodes(self.y)
        self._make_graph(self.y)
        self._resize_graph()
        return self._dot

    def _make_output_nodes(self, y):
        """"Adds output nodes into dot."""
        if isinstance(y, tuple):
            for yy in y:
                self._make_output_nodes(yy)
        elif isinstance(y, torch.Tensor):
            name = self._get_output_name.get_name(y)
            size = self._convert_size(y.size())
            node_name = '%s\n%s' % (name, size)
            self._dot.node(self._get_output_name.get_id(y), node_name,
                           fillcolor=OUTPUT_COLOR)
            self._visited.add(y)
        else:
            raise TypeError

    def _make_graph(self, y):
        """Adds the graph extracted from the output tensor(s)."""
        if isinstance(y, tuple):
            for yy in y:
                self._make_graph(yy)
        elif isinstance(y, torch.Tensor):
            self._add_nodes(y.grad_fn)
            self._add_edge(y.grad_fn, y)
        else:
            raise TypeError

    def _add_nodes(self, node):
        """Adds all nodes into dot."""
        if node not in self._visited:
            if hasattr(node, 'variable'):
                if self._is_param_node(node):
                    self._add_param_node(node)
                elif self._is_input_node(node):
                    self._add_input_node(node)
            else:
                self._add_op_node(node)
            self._visited.add(node)

            if hasattr(node, 'next_functions'):
                for subnode, _ in node.next_functions:
                    if subnode is not None:
                        self._add_edge(subnode, node)
                        self._add_nodes(subnode)
            if hasattr(node, 'saved_tensors'): # TODO
                print(node)

    def _add_edge(self, sub_node, parent_node):
        self._dot.edge(str(id(sub_node)), str(id(parent_node)))

    def _is_param_node(self, node):
        return self._get_param_name.has(node.variable)

    def _is_input_node(self, node):
        return self._get_input_name.has(node.variable)

    def _add_op_node(self, op):
        """Adds backward operation into graph."""
        op_id = str(id(op))
        name = str(type(op).__name__)
        self._dot.node(op_id, name, fillcolor=OP_COLOR)

    def _add_param_node(self, node):
        size = self._convert_size(node.variable.size())
        name = self._get_param_name.get_name(node.variable)
        node_id = str(id(node))
        color = PARAM_COLOR
        self._add_var_into_dot(node_id, name, size, color)

    def _add_input_node(self, node):
        size = self._convert_size(node.variable.size())
        name = self._get_input_name.get_name(node.variable)
        node_id = str(id(node))
        color = INPUT_COLOR
        self._add_var_into_dot(node_id, name, size, color)

    def _add_var_into_dot(self, node_id, name, size, color):
        node_name = '%s\n%s' % (name, size)
        self._dot.node(node_id, node_name, fillcolor=color)

    def _convert_input(self, x):
        """Sets ``requires_grad`` to True to show it in the graph."""
        if isinstance(x, tuple):
            for xx in x:
                self._convert_input(xx)
        elif isinstance(x, torch.Tensor):
            x.requires_grad = True
        else:
            raise TypeError
        return x

    def _calc_output(self, x, model):
        """Calculates the model output from input x."""
        if isinstance(x, tuple):
            return model(*x)
        else:
            return model(x)

    def _convert_size(self, size):
        """Converts the tensor size to str."""
        return tuple(size).__str__()

    def _resize_graph(self, size_per_element=0.15, min_size=12):
        """Resizes the graph according to how much content it contains."""
        num_rows = len(self._dot.body)
        content_size = num_rows * size_per_element
        size = max(min_size, content_size)
        size = ','.join([str(size)] * 2)
        self._dot.graph_attr.update(size=size)


class _GetName:
    def __init__(self):
        self._names = dict()
    def get_name(self, x):
        return self._names[self.get_id(x)]
    def get_id(self, x):
        return str(id(x))
    def has(self, x):
        return self.get_id(x) in self._names


class _GetParamName(_GetName):
    """Extracts the name of a parameter."""
    def __init__(self, model):
        self._names = {self.get_id(v): k for k, v in model.named_parameters()}


class _GetInputName(_GetName):
    """Extracts the name of an input tensor."""
    def __init__(self, x):
        self._names = _calc_name(x, 'input')


class _GetOutputName(_GetName):
    """Extracts the name of an output tensor."""
    def __init__(self, y):
        self._names = _calc_name(y, 'output')


def _calc_name(x, name='input', ind=''):
    """Calcuates the name of tensors."""
    names = dict()
    if isinstance(x, tuple):
        for i, xx in enumerate(x):
            names.update(_calc_name(xx, name, str(i)))
    elif isinstance(x, torch.Tensor):
        names[str(id(x))] = '%s%s' % (name, ind)
    else:
        raise TypeError
    return names
