import torch
import torch.nn as nn
from source.layers.E3DLSTM import E3DLSTMCell


class E3DLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, kernel_size, tau):
        super().__init__()

        self._tau = tau
        self._cells = []

        input_shape = list(input_shape)
        for i in range(num_layers):
            cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []

        for step, x in enumerate(input):
            for cell_idx, cell in enumerate(self._cells):
                if step == 0:
                    c_history, m, h = self._cells[cell_idx].init_hidden(
                        batch_size, self._tau, input.device
                    )
                    c_history_states.append(c_history)
                    h_states.append(h)

                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history, m, h = cell(
                    x, c_history_states[cell_idx], m, h_states[cell_idx]
                )
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h

            outputs.append(h)

        # NOTE Concat along the channels
        return torch.cat(outputs, dim=1)
