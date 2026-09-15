# -*- coding: utf-8 -*-
"""Wrap fig2_dcredit_architecture.svg into Inkscape layers for manual editing."""
import os
import re
import xml.etree.ElementTree as ET

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'fig2_dcredit_architecture.svg')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'fig2_dcredit_architecture_inkscape.svg')

SVG_NS = 'http://www.w3.org/2000/svg'
INK_NS = 'http://www.inkscape.org/namespaces/inkscape'
XLINK_NS = 'http://www.w3.org/1999/xlink'

ET.register_namespace('', SVG_NS)
ET.register_namespace('inkscape', INK_NS)
ET.register_namespace('xlink', XLINK_NS)
ET.register_namespace('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
ET.register_namespace('dc', 'http://purl.org/dc/elements/1.1/')
ET.register_namespace('cc', 'http://creativecommons.org/ns#')

text = open(SRC, encoding='utf-8').read()
text = re.sub(r'<!DOCTYPE[^>]*>', '', text, flags=re.S)

root = ET.fromstring(text)


def find_id(el, target):
    if el.get('id') == target:
        return el
    for child in el:
        found = find_id(child, target)
        if found is not None:
            return found
    return None


axes = find_id(root, 'axes_1')
if axes is None:
    raise SystemExit('axes_1 group not found')


def classify(el):
    eid = el.get('id') or ''
    if eid.startswith('box-'):
        return 'Boxes'
    if eid.startswith('arrow-'):
        return 'Arrows'
    if eid.startswith('label-'):
        return 'Labels'
    return 'Other'


children = list(axes)
layers = {}
order = []
for el in children:
    name = classify(el)
    if name not in layers:
        layers[name] = []
        order.append(name)
    layers[name].append(el)

for el in children:
    axes.remove(el)

for name in order:
    g = ET.SubElement(axes, f'{{{SVG_NS}}}g')
    g.set('id', 'layer-' + name.lower())
    g.set(f'{{{INK_NS}}}groupmode', 'layer')
    g.set(f'{{{INK_NS}}}label', name)
    for el in layers[name]:
        g.append(el)

out = ET.tostring(root, encoding='unicode', xml_declaration=False)
out = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + out + '\n'
open(DST, 'w', encoding='utf-8').write(out)
print('layers written:', order)
print('saved ->', DST)
