#!/usr/bin/env python

"""
Biomorph Evolve - Implementation of Dawkins' Blind Watchmaker Algorithm
Copyright (C) 2017 James Garnon

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Biomorph Evolve version 1.1
Download Site: https://gatc.ca
"""

# __pragma__ ('skip')
from __future__ import division
# __pragma__ ('noskip')
import random
import numpy as np
from control import App, Control, Config, DictCache, ListCache, RectCache, Renderer, Color

if not hasattr(random, 'randrange'):
    random.randrange = lambda i, f: random.choice(range(i, f))

size = 200
col = 3
row = 3
gene = {'num': 9, 'lv': (-9, 9), 'ln': (1, 10), 'cl': (0, 256), 'jmp': (1, 4)}
genome = dict([(i, gene['lv']) for i in range(1, gene['num'] + 1)])
for i in range(10, 13):
    genome[i] = gene['cl']
for i in range(13, 18):
    genome[i] = gene['jmp']
for i in range(18, 25):
    genome[i] = gene['ln']
biomorph_color = Color(40, 60, 80)
screen_color = Color(150, 150, 150)
grid_color = Color(0, 20, 40)
dict_cache = DictCache()
list_cache = ListCache()
rect_cache = RectCache()


class Matrix(object):

    def __init__(self):
        self.biomorph_selected = None
        self.pos_select = None
        self.biomorph = []
        self.processing = None
        self.reset = False
        self.dx = [0 for _ in range(9)]
        self.dy = [0 for _ in range(9)]
        self.repeat = False
        self.control = None
        self.size = size
        self.color = [0 for _ in range(3)]
        self.jmp = [0 for _ in range(5)]
        self.ln = np.array([0 for _ in range(7)])
        self.screen_color = screen_color
        self.grid_color = grid_color
        self.renderer = Renderer(self)
        self.pos = [(c * size + (size // 2), r * size + (size // 2)) for r in range(row) for c in range(col)]
        self.grid = [(c * size, r * size) for r in range(row) for c in range(col)]
        self.rect = [rect_cache.get(grid[0], grid[1], size, size) for grid in self.grid]
        self.renderer.draw_grid(self.grid, (size, size), self.grid_color)
        self.init()
        self.renderer.update()

    def init(self):
        for i in range((row * col)):
            self.biomorph.append(Biomorph())
        for i, biomorph in enumerate(self.biomorph):
            self.develop(biomorph)
            self.display(biomorph, self.pos[i])

    def prune(self):
        for biomorph in self.biomorph:
            if not self.biomorph_selected:
                biomorph.deinit()
            else:
                if biomorph.segments is not None:
                    biomorph.segments.idx = 0
        self.biomorph[:] = []

    def reproduce(self, biomorph):
        for i in range((row * col) - 1):
            self.biomorph.append(biomorph.reproduce())
        self.biomorph.insert(4, biomorph)
        return self.biomorph

    def develop(self, biomorph):
        startx = 0
        starty = 0
        startdir = 2
        biomorph.develop(startx, starty, startdir, self.dx, self.dy, self.color, self.jmp, self.ln)

    def display(self, biomorph, pos):
        self.renderer.render(biomorph, pos)

    def biomorph_select(self, pos):
        self.pos_select = pos
        for i, r in enumerate(self.rect):
            if r.collidepoint(pos):
                self.biomorph_selected = self.biomorph[i]
                break
        return self.biomorph_selected

    def restart(self):
        self.renderer.clear()
        self.prune()
        self.init()
        self.renderer.update()

    def refresh(self):
        self.renderer.update()

    def terminate(self):
        self.renderer.blank()
        self.prune()

    def update(self):
        if self.biomorph_selected is not None:
            if not self.reset:
                self.control.set_wait(True)
                self.renderer.clear()
                self.prune()
                self.reset = True
                self.processing = 0
                self.biomorph = self.reproduce(self.biomorph_selected)
            else:
                if self.processing < (row * col):
                    biomorph = self.biomorph[self.processing]
                    self.develop(biomorph)
                    self.display(biomorph, self.pos[self.processing])
                    self.processing += 1
                else:
                    self.renderer.update()
                    if self.repeat:
                        self.biomorph_selected = self.biomorph_selected.reproduce()
                        self.reset = False
                    else:
                        self.biomorph_selected = None
                        self.control.set_wait(False)
                        self.reset = False


class Biomorph(object):

    def __init__(self, genes=None):
        self.genes = dict_cache.get()
        if genes is not None:
            for i in genes.keys():
                self.genes[i] = genes[i]
            i = random.choice(list(self.genes.keys()))
            if i < 10 or i >= 13:
                self.genes[i] += random.choice([-1, 1])
            else:
                self.genes[i] += random.randrange(0, 256)
                self.genes[i] %= 256
            if self.genes[i] < genome[i][0]:
                self.genes[i] = genome[i][0] + 1
            elif self.genes[i] > genome[i][1]:
                self.genes[i] = genome[i][1] - 1
        else:
            for i in range(1, 10):
                self.genes[i] = random.randrange(genome[i][0], genome[i][1] + 1)
            self.genes[9] = random.randrange(genome[i][1] - 3, genome[i][1] + 1)
            for i in range(10, 13):
                self.genes[i] = random.randrange(genome[i][0], genome[i][1])
            for i in range(13, 25):
                self.genes[i] = random.randrange(genome[i][0], genome[i][1] + 1)
        self.segments = None
        self.rect = rect_cache.get(0, 0, size, size)
        self.dim = size

    def deinit(self):
        dict_cache.set(self.genes)
        rect_cache.set(self.rect)
        self.segments.deinit()

    def reproduce(self):
        return Biomorph(self.genes)

    def develop(self, startx, starty, startdir, dx, dy, color, jmp, len):
        self.segments = Segment()
        len, dx, dy, color, jmp = self.plugin(self.genes, dx, dy, color, jmp, len)
        self.tree(startx, starty, len, 4, startdir, dx, dy, color, jmp, 2)

    def tree(self, x, y, len, len_p, dir, dx, dy, color, jmp, jmp_p):
        if len_p < 0:
            len_p = 6
        if len_p > 6:
            len_p = 0
        _dir = dir % 8
        xnew = x + len[len_p] * dx[_dir]
        ynew = y + len[len_p] * dy[_dir]
        self.segments.add(x, y, xnew, ynew, color)
        if jmp_p < 0:
            jmp_p = 4
        if jmp_p > 4:
            jmp_p = 0
        if len[len_p] > 0:
            self.tree(xnew, ynew, len - 1, len_p - 1, _dir + jmp[jmp_p], dx, dy, self.rotate_color(color, True), jmp, jmp_p + 1)
            self.tree(xnew, ynew, len - 1, len_p + 1, _dir + jmp[jmp_p], dx, dy, self.rotate_color(color, False), jmp, jmp_p - 1)

    def rotate_color(self, color, sense):
        if sense:
            return [color[2], color[0], color[1]]
        else:
            return [color[1], color[2], color[0]]

    def plugin(self, gene, dx, dy, color, jmp, len):
        dx[3] = gene[1]
        dx[4] = gene[2]
        dx[5] = gene[3]
        dx[1] = -dx[3]
        dx[0] = -dx[4]
        dx[2] = 0
        dx[6] = 0
        dx[7] = -dx[5]
        dy[2] = gene[4]
        dy[3] = gene[5]
        dy[4] = gene[6]
        dy[5] = gene[7]
        dy[6] = gene[8]
        dy[0] = dy[4]
        dy[1] = dy[3]
        dy[7] = dy[5]

        color[0] = abs(gene[10]) % 256
        color[1] = abs(gene[11]) % 256
        color[2] = abs(gene[12]) % 256

        jmp[0] = gene[13]
        jmp[1] = gene[14]
        jmp[2] = gene[15]
        jmp[3] = gene[16]
        jmp[4] = gene[17]

        len[0] = gene[18]
        len[1] = gene[19]
        len[2] = gene[20]
        len[3] = gene[21]
        len[4] = gene[22]
        len[5] = gene[23]
        len[6] = gene[24]

        return len, dx, dy, color, jmp


class Segment(object):
    _cache = []

    def __init__(self):
        self.colors = self.get_list() #
        self.x1 = self.get_list()
        self.y1 = self.get_list()
        self.x2 = self.get_list()
        self.y2 = self.get_list()
        self.idx = 0
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0

    def add(self, x1, y1, x2, y2, color):
        self.x1[self.idx] = x1
        self.y1[self.idx] = y1
        self.x2[self.idx] = x2
        self.y2[self.idx] = y2
        self.colors[self.idx] = Color(color[0], color[1], color[2])
        self.idx += 1
        return None

    def dim(self):
        self.xmin = min(self.x2)
        self.xmax = max(self.x2)
        self.ymin = min(self.y2)
        self.ymax = max(self.y2)
        return self.xmax - self.xmin, self.ymax - self.ymin

    def transform(self, size, pos):
        width, height = self.dim()
        maxsize = max(width, height)
        _size = size - 6
        if _size > maxsize:
            adj = 1
        else:
            adj = maxsize / _size
        mid = _size // 2
        xmin = self.xmin // adj
        xmax = self.xmax // adj
        ymin = self.ymin // adj
        ymax = self.ymax // adj
        w = xmax - xmin
        h = ymax - ymin
        x = pos[0] - xmin - (w // 2) - 1
        y = pos[1] - ymin - (h // 2) - 1
        return x, y, adj

    def get_list(self):
        if len(self._cache) > 0:
            return self._cache.pop()
        else:
            return [0 for i in range(2 ** (gene['ln'][1] + 1))]

    def deinit(self):
        for i in range(self.idx):
            self.x1[i] = 0
            self.y1[i] = 0
            self.x2[i] = 0
            self.y2[i] = 0
        self._cache.append(self.x1)
        self._cache.append(self.y1)
        self._cache.append(self.x2)
        self._cache.append(self.y2)
        self.idx = 0
        return None


app = matrix = control = None


def run():
    matrix.update()
    quit = control.update()
    if quit:
        matrix.terminate()
        app.terminate()


def main():
    global app, matrix, control
    config = Config()
    config.setup(size * col, size * row)
    matrix = Matrix()
    control = Control(matrix)
    matrix.control = control
    app = App(run)
    app.run()


if __name__ == '__main__':
    main()
