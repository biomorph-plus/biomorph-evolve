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
gene = {
    'lv': {'num': 16, 'r': (-24, 24)},
    'cl': {'num': 3, 'r': (0, 256)},
    'jmp': {'num': 8, 'r': (1, 8)},
    'ln': {'num': 8, 'r': (3, 7)},
    'fk': {'num': 4, 'r': (1, 5)}
}
num = gene['lv']['num'] + 1
genome = dict([(i, gene['lv']['r']) for i in range(1, num)])
for i in range(num, num + gene['cl']['num']):
    genome[i] = gene['cl']['r']
num += gene['cl']['num']
for i in range(num, num + gene['jmp']['num']):
    genome[i] = gene['jmp']['r']
num += gene['jmp']['num']
for i in range(num, num + gene['ln']['num']):
    genome[i] = gene['ln']['r']
num += gene['ln']['num']
for i in range(num, num + gene['fk']['num']):
    genome[i] = gene['fk']['r']
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
        self.dx = np.zeros(gene['lv']['num'] // 2)
        self.dy = np.zeros(gene['lv']['num'] // 2)
        self.repeat = False
        self.control = None
        self.size = size
        self.color = [0 for _ in range(gene['cl']['num'])]
        self.jmp = [0 for _ in range(gene['jmp']['num'])]
        self.ln = np.zeros(gene['ln']['num'])
        self.fk = [0 for _ in range(gene['fk']['num'])]
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
        biomorph.develop(startx, starty, startdir, self.dx, self.dy, self.color, self.jmp, self.ln, self.fk)

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
            num_ = gene['lv']['num'] + 1
            if num_ < i <= num_ + gene['cl']['num']:
                self.genes[i] += random.randrange(0, 256)
                self.genes[i] %= 256
            else:
                self.genes[i] += random.choice([-1, 1])
            if self.genes[i] < genome[i][0]:
                self.genes[i] = genome[i][0] + 1
            elif self.genes[i] > genome[i][1]:
                self.genes[i] = genome[i][1] - 1
        else:
            num_ = gene['lv']['num'] + 1
            for i in range(1, num_):
                self.genes[i] = random.randrange(genome[i][0], genome[i][1] + 1)
            for i in range(num_, num_ + gene['cl']['num']):
                self.genes[i] = random.randrange(genome[i][0], genome[i][1] + 1)
            num_ += gene['cl']['num']
            for i in range(num_, num_ + gene['jmp']['num']):
                self.genes[i] = random.randrange(genome[i][0], genome[i][1])
            num_ += gene['jmp']['num']
            for i in range(num_, num_ + gene['ln']['num']):
                self.genes[i] = random.randrange(genome[i][0], genome[i][1] + 1)
            num_ += gene['ln']['num']
            for i in range(num_, num_ + gene['fk']['num']):
                self.genes[i] = random.randrange(genome[i][0], genome[i][1] + 1)
        self.segments = None
        self.idx = 0
        self.rect = rect_cache.get(0, 0, size, size)
        self.dim = size

    def deinit(self):
        dict_cache.set(self.genes)
        rect_cache.set(self.rect)
        self.segments.deinit()

    def reproduce(self):
        return Biomorph(self.genes)

    def develop(self, startx, starty, startdir, dx, dy, color, jmp, len, fk):
        len, dx, dy, color, jmp, fk = self.plugin(dx, dy, color, jmp, len, fk)
        self.pretree(len, 0, startdir, jmp, 0, fk, 0)
        self.segments = Segment(self.idx)
        self.tree(startx, starty, len, 0, startdir, dx, dy, color, jmp, 0, fk, 0)

    def tree(self, x, y, len, len_p, dir, dx, dy, color, jmp, jmp_p, fk, fk_p):
        if len_p < 0:
            len_p = gene['ln']['num']
        if len_p >= gene['ln']['num']:
            len_p = 0
        if fk_p < 0:
            fk_p = gene['fk']['num']
        if fk_p >= gene['fk']['num']:
            fk_p = 0
        _dir = dir % gene['lv']['num'] // 2
        xnew = x + len[len_p] * dx[_dir]
        ynew = y + len[len_p] * dy[_dir]
        self.segments.add(x, y, xnew, ynew, color)
        if jmp_p < 0:
            jmp_p = gene['jmp']['num']
        if jmp_p >= gene['jmp']['num']:
            jmp_p = 0
        if len[len_p] > 0:
            for i in range(fk[fk_p]):
                i_ = i - fk[fk_p] // 2
                self.tree(xnew, ynew, len - 1, len_p + i_, _dir + jmp[jmp_p], dx + i_, dy + i_, self.rotate_color(color, i_), jmp, jmp_p + i_, fk, fk_p + i_)

    def pretree(self, len, len_p, dir, jmp, jmp_p, fk, fk_p):
        self.idx += 1
        if len_p < 0:
            len_p = gene['ln']['num']
        if len_p >= gene['ln']['num']:
            len_p = 0
        if fk_p < 0:
            fk_p = gene['fk']['num']
        if fk_p >= gene['fk']['num']:
            fk_p = 0
        _dir = dir % gene['lv']['num'] // 2
        if jmp_p < 0:
            jmp_p = gene['jmp']['num']
        if jmp_p >= gene['jmp']['num']:
            jmp_p = 0
        if len[len_p] > 0:
            for i in range(fk[fk_p]):
                i_ = i - fk[fk_p] // 2
                self.pretree(len - 1, len_p + i_, _dir + jmp[jmp_p], jmp, jmp_p + i_, fk, fk_p + i_)

    def rotate_color(self, color, sense):
        color_ = [0, 0, 0]
        if sense < 0:
            for _ in range(-sense):
                color_ = [color[2], color[0], color[1]]
                color = color_
        else:
            for _ in range(sense):
                color_ = [color[1], color[2], color[0]]
                color = color_
        return color

    def plugin(self, dx, dy, color, jmp, len, fk):
        num_ = 1
        for i in range(int(gene['lv']['num'] / 2)):
            dy[i] = self.genes[num_ + i]

        num_ += int(gene['lv']['num'] / 2)
        for i in range(int(gene['lv']['num'] / 2)):
            dx[i] = self.genes[num_ + i]

        num_ += int(gene['lv']['num'] / 2)
        for i in range(gene['cl']['num']):
            color[i] = abs(self.genes[num_ + i]) % 256

        num_ += gene['cl']['num']
        for i in range(gene['jmp']['num']):
            jmp[i] = self.genes[num_ + i]

        num_ += gene['jmp']['num']
        for i in range(gene['ln']['num']):
            len[i] = self.genes[num_ + i]
        num_ += gene['ln']['num']
        for i in range(gene['fk']['num']):
            fk[i] = self.genes[num_ + i]

        return len, dx, dy, color, jmp, fk


class Segment(object):
    _cache = []

    def __init__(self, midx):
        self.midx = midx
        self.colors = self.get_list()
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
            return [0 for _ in range(self.midx)]

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
