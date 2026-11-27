"""Render the exact four-completion assignment example; no empirical inputs."""
from pathlib import Path
import re, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import fitz
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def main(output):
    output = Path(output)
    (output / "sources").mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    green, ink, gray = '#276657', '#20252a', '#616a71'
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'path'})
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.35))
    fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.02, wspace=.12)
    A = np.array([1., 1., -1., -1.])
    h = np.array([[1., 1.], [1., 1.], [1., 0.], [0., 1.]])
    states = []
    for j, ax in enumerate(axes):
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        ax.text(.5, .98, '(a) Original assignment' if j == 0 else '(b) Penalties swapped',
                ha='center', va='top', fontsize=10, weight='bold', color=ink)
        q = np.array([0., 0., 1., 3.] if j == 0 else [0., 0., 3., 1.])
        reward = np.array([1., 1., 0., 0.]) - q
        B = (reward-reward.mean()) / reward.std()
        alpha = float(A @ B / (A @ A)); W = B-alpha*A; S = W@h/4
        assert np.allclose(W[:2], 0) and np.isclose(alpha, 3/math.sqrt(11))
        assert np.isclose(1-alpha**2, 2/11)
        for i, x in [(2, .04), (3, .54)]:
            ax.add_patch(FancyBboxPatch((x, .42), .42, .43,
                         boxstyle='round,pad=.01,rounding_size=.02',
                         facecolor='#f5f6f7', edgecolor='#a9b0b4', linewidth=.7))
            ax.text(x+.21, .78, f'$y_{i+1}$: incorrect', ha='center', fontsize=9)
            ax.text(x+.21, .67, f'$q_{i+1}={int(q[i])}$', ha='center', fontsize=11)
            sign = '+' if W[i] > 0 else '-'
            ax.text(x+.21, .55, rf'$W_{i+1}={sign}2/\sqrt{{11}}$', ha='center', fontsize=10, color=green)
            ax.text(x+.21, .32, rf'$h_{i+1}=({int(h[i,0])},{int(h[i,1])})$', ha='center', fontsize=9)
            ax.annotate('', xy=(x+.21,.37), xytext=(x+.21,.41),
                        arrowprops={'arrowstyle':'->','color':gray,'lw':.8})
        # Both panels use identical axes and scale; only residual weights move.
        ox, oy, scale = .17, .13, .55
        ax.plot([ox-.08,ox+.10],[oy,oy],color='#b0b7bc',lw=.6)
        ax.plot([ox,ox],[oy-.10,oy+.10],color='#b0b7bc',lw=.6)
        ax.annotate('', xy=(ox+scale*S[0],oy+scale*S[1]), xytext=(ox,oy),
                    arrowprops={'arrowstyle':'->','color':green,'lw':1.6})
        numerator = '(1,-1)' if j==0 else '(-1,1)'
        ax.text(.62, .14, rf'$S=\frac{{{numerator}}}{{2\sqrt{{11}}}}$',ha='center',va='center',fontsize=12,color=green)
        states.append({'penalties':q.tolist(),'advantages':B.tolist(),'residual':W.tolist(),
                       'residual_gradient':S.tolist(),'cosine':alpha,'energy':1-alpha**2})
    assert np.allclose(states[0]['residual_gradient'], -np.array(states[1]['residual_gradient']))
    dest=output; svg=dest/'sources/assignment_example.svg'; pdf=dest/'assignment_example.pdf'
    fig.savefig(svg)
    fig.savefig(dest/'previews/assignment_example.png', dpi=200)
    plt.close(fig)
    renderPDF.drawToFile(svg2rlg(str(svg)),str(pdf))
    doc=fitz.open(pdf); page=doc[0]; before=page.get_pixmap().samples
    stream=page.read_contents()
    for block in re.findall(rb'BT\b.*?\bET',stream,re.S):
        assert not re.search(rb'\bTj\b|\bTJ\b',block)
    stream=re.sub(rb'BT\b.*?\bET',b'',stream,flags=re.S)
    page.clean_contents();doc.update_stream(page.get_contents()[0],stream)
    kind,res=doc.xref_get_key(page.xref,'Resources')
    if kind=='xref':doc.xref_set_key(int(res.split()[0]),'Font','<<>>')
    else:doc.xref_set_key(page.xref,'Resources/Font','<<>>')
    assert page.get_pixmap().samples==before
    data=doc.tobytes(garbage=4,deflate=True);doc.close();pdf.write_bytes(data)
    (output/'assignment_example_validation.json').write_text(json.dumps({'status':'PASS','states':states},indent=2)+'\n')
    print('Built vector assignment example; exact toy arithmetic and rendering verified.')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('regenerated-figures-assignment'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output folder.')
    main(args.output)
