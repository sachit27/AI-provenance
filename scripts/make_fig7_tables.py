#!/usr/bin/env python3
"""Build Figure 7 and the supplementary table fragments from saved analysis outputs.

See analyses/README.md for the inputs and commands.
"""
import os, re, json, math, glob
import numpy as np, pandas as pd

import argparse
from pathlib import Path
_HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument('--root', type=Path, default=_HERE.parent / 'analyses', help='directory containing the analysis outputs')
_ap.add_argument('--extra-root', default=None, help='optional second directory searched for input files with the same relative layout')
_ap.add_argument('--build-dir', type=Path, default=_HERE.parent / 'outputs' / 'paper', help='directory for the generated figure and table fragments')
_args = _ap.parse_args()
REV = _args.root
INP = _args.extra_root or REV
OUT = _args.build_dir
os.makedirs(OUT, exist_ok=True)
MISSING = []

def inp(*p):
    """locate file in staged revision dir or extracted archive"""
    for root in (REV, INP):
        f = os.path.join(root, *p)
        if os.path.exists(f): return f
    return None

# ------------------------------------------------------------------ helpers
def esc(s):
    s = str(s)
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'),
                 ('$', r'\$'), ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')]:
        s = s.replace(a, b)
    return s
def f3(x): return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'
def f1(x): return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.1f}'
def sg(x):  # signed 3 decimals with proper minus
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else (f'{x:.3f}'.replace('-', '$-$') if x < 0 else f'{x:.3f}')
def fmtp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return '--'
    p = float(p)
    if p >= 0.001: return f'{p:.2g}'
    e = int(math.floor(math.log10(p))); m = float(f'{p/10**e:.2g}')
    if m >= 10: m /= 10; e += 1
    return rf'${m:.1f}\times10^{{{e}}}$'
def fmtp_str(s):  # p given as text like 1.4e-06
    return fmtp(float(s))
def wilson(k, n, z=1.959964):
    ph = k / n; den = 1 + z*z/n
    c = (ph + z*z/(2*n)) / den
    h = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    return c-h, c+h
def table(body, cap, label, cols, head=None, wide=False, small='\\small', tight=False):
    cap = cap.replace("\\'", "'")
    hd = f'\\toprule\n{head}\\\\\n\\midrule\n' if head else ''
    tc = '\\setlength{\\tabcolsep}{3pt}\n' if tight else ''
    inner = tc + f'\\begin{{tabular}}{{{cols}}}\n{hd}{body}\\bottomrule\n\\end{{tabular}}'
    if wide: inner = '\\resizebox{\\linewidth}{!}{' + inner + '}'
    return (f'\\begin{{table}}[H]\n\\revon\n\\centering{small}\n\\caption{{{cap}}}\n\\label{{{label}}}\n{inner}\n\\end{{table}}\n')
def longtab(body, cap, label, cols, head, small='\\footnotesize', ncol=None):
    cap = cap.replace("\\'", "'")
    ncol = ncol or head.count('&') + 1
    hd = f'\\toprule\n{head}\\\\\n\\midrule\n' if head else '\\toprule\n'
    return ('{\\revon\n' + small + '\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{' + cols + '}\n'
            f'\\caption{{{cap}}}\\label{{{label}}}\\\\\n{hd}\\endfirsthead\n'
            f'\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Table \\thetable{{}} (continued)}}}}\\\\\n{hd}\\endhead\n'
            f'\\midrule\n\\multicolumn{{{ncol}}}{{r}}{{\\textit{{continued on next page}}}}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n'
            + body + '\\end{longtable}\n}\n')

TOPICS = ['education', 'trust', 'q7', 'q21']
TOPIC_NAME = {'education': 'Education', 'trust': 'Trust', 'q7': 'Scotland Q7', 'q21': 'Scotland Q21'}
MODELS = ['gpt-4o-mini', 'qwen3.6-35b-a3b', 'glm-4.7-flash']
MODEL_TEX = {'gpt-4o-mini': 'GPT-4o-mini', 'qwen3.6-35b-a3b': 'Qwen3.6-35B-A3B', 'glm-4.7-flash': 'GLM-4.7-flash'}

# ------------------------------------------------------------------ parse llm_summaries/RESULTS.md
txt = open(inp('llm_summaries', 'RESULTS.md'), encoding='utf8').read()
def md_table(block):
    rows = []
    for l in block.strip().split('\n'):
        if l.startswith('|'): rows.append(l)
        elif rows: break
    hdr = [c.strip() for c in rows[0].strip('|').split('|')]
    out = []
    for r in rows[2:]:
        out.append([c.strip() for c in r.strip('|').split('|')])
    return hdr, out
def section(text, start, end=None):
    i = text.index(start) + len(start)
    j = text.index(end, i) if end and end in text[i:] else len(text)
    return text[i:j]

budget_hdr, budget_rows = md_table(section(txt, '## Length and budget compliance', '## Coverage and structure'))
LLM = {}  # (topic, emb) -> dict
for t in TOPICS:
    for emb in ['openai', 'mpnet']:
        m = re.search(rf'### {t} — embedding `{emb}`.*?(?=\n### |\n## |\Z)', txt, re.S)
        blk = m.group(0)
        d = {}
        h, r = md_table(blk[blk.index('| summarizer | n_seeds'):])
        d['cov'] = {x[0]: x for x in r}
        sp = blk[blk.index('Spearman correlation of participant'):]
        h, r = md_table(sp[sp.index('|'):])
        d['spear'] = {x[0]: dict(zip(h[1:], x[1:])) for x in r}
        w = re.search(r'Within-model seed-to-seed Spearman: (.*)', blk).group(1)
        d['within'] = {a.split(':')[0].strip(): float(a.split(':')[1]) for a in w.split(';')}
        jb = blk[blk.index('Jaccard overlap'):]
        h, r = md_table(jb[jb.index('|'):])
        d['jac'] = {x[0]: x for x in r}
        k = re.search(r"Kendall's W across summarizers on region coverage: all \(incl\. official\) W=([\d.]+) \(perm p=([\d.]+)\); LLMs only W=([\d.]+) \(p=([\d.]+)\)\. Lowest-covered region per summarizer: (.*)", blk)
        d['W'] = k.groups()
        low = [x.split('→')[1].strip().rstrip('.') for x in k.group(5).split(', ')]
        d['lowest'] = low
        d['nreg'] = int(re.search(r'cluster_id, (\d+) regions\)', blk).group(1))
        d['n'] = re.search(r'n=(\d+)', blk).group(1)
        LLM[(t, emb)] = d

# ------------------------------------------------------------------ FIGURE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 6.8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
OI = {'blue': '#0072B2', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73', 'verm': '#D55E00', 'purple': '#CC79A7', 'grey': '#666666'}
fig, axs = plt.subplots(2, 2, figsize=(7, 5.2), dpi=300)
fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.115, wspace=0.30, hspace=0.42)
for ax, L in zip(axs.flat, 'abcd'):
    ax.text(-0.17, 1.06, L, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')

# panel a ---------------------------------------------------------
def panel_a(ax):
    df = pd.read_csv(inp('scotland_analysis', 'scotland_analysis_summary.csv'))
    df = df[df.embedding_model == 'text-embedding-3-large'].set_index('topic')
    qs = ['q4', 'q7', 'q20', 'q21', 'q22', 'q23']; x = np.arange(len(qs)); off = 0.13
    for key, mk, col, lab, dx in [('mean_opt', 'o', OI['blue'], 'Mean-optimized (mean coverage gain)', -off), ('tail_opt', '^', OI['orange'], 'Tail-oriented (bottom-decile gain)', off)]:
        y = df.loc[qs, key + '_gain'].values; lo = df.loc[qs, key + '_gain_ci_low'].values; hi = df.loc[qs, key + '_gain_ci_high'].values
        ax.errorbar(x + dx, y, yerr=[y - lo, hi - y], fmt=mk, color=col, ms=4.5, lw=0.8, capsize=2, capthick=0.8, label=lab, zorder=3)
    ax.axhline(0, color='k', lw=0.7, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([q.upper() for q in qs]); ax.set_xlabel('Scottish consultation question')
    ax.set_ylabel('Held-out gain over official text'); ax.set_ylim(-0.01, 0.17)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.30), fontsize=6.2)
    ax.grid(axis='y', lw=0.3, alpha=0.5, zorder=0)

# panel b ---------------------------------------------------------
def panel_b(ax):
    cols = [OI['blue'], OI['orange'], OI['green']]
    TB = ['education', 'trust', 'q7']   # Q21 is excluded from the controlled comparison
    w = 0.25; x0 = np.arange(len(TB))
    for i, m in enumerate(MODELS):
        v = [float(LLM[(t, 'openai')]['spear']['official'][m]) for t in TB]
        ax.bar(x0 + (i-1)*w, v, w*0.92, color=cols[i], label=MODEL_TEX[m], zorder=2)
        wi = [LLM[(t, 'openai')]['within'][m] for t in TB]
        ax.scatter(x0 + (i-1)*w, wi, marker='D', s=9, facecolor='white', edgecolor='k', linewidth=0.6, zorder=3)
        for xx, vv in zip(x0 + (i-1)*w, v):
            ax.text(xx, 0.41, f'{vv:.2f}'.lstrip('0'), ha='center', va='bottom', fontsize=5.6, color='white', rotation=90, zorder=4)
    ax.set_xticks(x0); ax.set_xticklabels(['Education', 'Trust', 'Scot. Q7'])
    ax.set_ylim(0.4, 1.0); ax.set_ylabel('Spearman correlation, official vs\nmodel-average coverage')
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker='D', ls='', mfc='white', mec='k', ms=3.5, mew=0.6)); l.append('Seed-to-seed (same model)')
    ax.legend(h, l, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=2, handletextpad=0.4, columnspacing=0.9, fontsize=6.2)
    ax.grid(axis='y', lw=0.3, alpha=0.5, zorder=0)

# panel c ---------------------------------------------------------
def panel_c(ax):
    f = inp('scotland_analysis', 'analysis', 'q7', 'participant_coverage.csv')
    if f is None:
        MISSING.append('fig7c (analysis/q7/participant_coverage.csv)'); ax.text(.5, .5, 'data missing', transform=ax.transAxes, ha='center'); return
    cov = pd.read_csv(f, dtype={'respondent_id': str}); cov = cov[cov.embedding_model == 'openai']
    g = pd.read_csv(inp('stance_groups', 'respondent_groups.csv'), dtype={'respondent_id': str})
    d = cov.merge(g[['respondent_id', 'g_q7_Q5']], on='respondent_id')
    globals()['PANEL_C'] = d
    data = [d[d.g_q7_Q5 == k].coverage.values for k in ['support', 'oppose']]
    labs = [f'Support\n(n={len(data[0])})', f'Oppose\n(n={len(data[1])})']
    bp = ax.boxplot(data, widths=0.5, showfliers=True, patch_artist=True, flierprops=dict(marker='.', ms=2, mfc='#999', mec='none', alpha=.6),
                    medianprops=dict(color='k', lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for p, c in zip(bp['boxes'], [OI['sky'], OI['verm']]): p.set_facecolor(c); p.set_alpha(0.75)
    ax.scatter([1, 2], [x.mean() for x in data], marker='D', s=22, c='white', edgecolors='k', linewidths=0.8, zorder=5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(labs)
    ax.set_ylabel('Coverage of own response\n(official summary)'); ax.set_xlabel('Own answer to Q5')
    ax.legend(handles=[Line2D([], [], marker='D', ls='', mfc='white', mec='k', ms=4.5, label='Mean')], frameon=False, loc='lower right')
    globals()['C_DIFF'] = data[1].mean() - data[0].mean()

# panel d ---------------------------------------------------------
_stf = inp('idea_audit', 'stance_test_results.csv')
assert _stf, 'idea_audit/stance_test_results.csv not found (run idea_audit/stance_test.py)'
STANCE = pd.read_csv(_stf).to_dict('records')
assert len(STANCE) == 8, len(STANCE)
def stance_row(minr, summ, scope):
    return [r for r in STANCE if r['min_resp'] == minr and r['summaries'] == summ and r['scope'] == scope][0]
def boot_generated(B=2000, seed=1):
    ids = []
    for t in ['education', 'trust']:
        wd = inp('idea_audit', 'work', t)
        i = pd.read_parquet(os.path.join(wd, 'ideas_t0.30.parquet'))[['idea_id', 'stance', 'n_resp']]
        j = pd.read_parquet(os.path.join(wd, 'judgments_t0.30.parquet'))
        j = j[(j.summary != 'official') & (~j.summary.str.startswith('official'))]
        sh = j.assign(m=(j.label == 'MISSED').astype(float)).groupby('idea_id').agg(share=('m', 'mean'), ns=('m', 'size')).reset_index()
        assert (sh.ns == 9).all()
        i = i[i.n_resp >= 10].merge(sh, on='idea_id'); i['topic'] = t
        ids.append(i)
    d = pd.concat(ids, ignore_index=True)
    d['grp'] = np.where(d.stance.isin(['oppose', 'concern']), 'opp', 'oth')
    rng = np.random.default_rng(seed); out = {}
    for g in ['opp', 'oth']:
        v = d[d.grp == g].share.values
        pt = 100*v.mean()
        bs = [100*v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
        out[g] = (pt, *np.percentile(bs, [2.5, 97.5]), len(v))
    return out
BOOT = boot_generated()
print('bootstrap (pooled generated missed %, lo, hi, n ideas):', BOOT)
assert abs(BOOT['opp'][0]-67.2) < 0.1 and abs(BOOT['oth'][0]-35.6) < 0.1

def panel_d(ax):
    rows = {sm: stance_row(10, sm, 'Canada') for sm in ['official', 'generated']}
    cols = {'opp': OI['verm'], 'oth': OI['sky']}
    w = 0.34
    for j, (key, lab) in enumerate([('opp', 'Oppose / concern'), ('oth', 'Other')]):
        xs, ys, lo, hi = [], [], [], []
        for i, sm in enumerate(['official', 'generated']):
            r = rows[sm]; k = int(r['k_' + key]); n = int(r['n_' + key])
            if sm == 'official':
                a, b = wilson(k, n); a, b = 100*a, 100*b; pt = 100*k/n; lab_n = f'(n={n})'
            else:
                pt, a, b, ni = BOOT[key]; lab_n = f'({ni} ideas)'
            xs.append(i + (j-0.5)*w); ys.append(pt); lo.append(pt-a); hi.append(b-pt)
            ax.text(xs[-1], b + 1.5, f'{pt:.1f}%\n{lab_n}', ha='center', va='bottom', fontsize=6)
        ax.bar(xs, ys, w*0.92, color=cols[key], label=lab, zorder=2, yerr=[lo, hi], error_kw=dict(lw=0.7, capsize=2, capthick=0.7))
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Official summaries', 'Generated summaries'])
    ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel('Recurring ideas judged missed (%)')
    ax.legend(frameon=False, loc='upper center', ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.04))
    ax.grid(axis='y', lw=0.3, alpha=0.5, zorder=0)

panel_b(axs[0, 1]); panel_d(axs[1, 1]); panel_c(axs[1, 0]); panel_a(axs[0, 0])
fig.savefig(os.path.join(OUT, 'fig7_extensions.png'), dpi=300)
print('fig saved; missing:', MISSING)


# =================================================================== TABLES
T = {}
EMB_TEX = {'openai': 'OpenAI', 'mpnet': 'MPNet', 'nomic': 'Nomic'}
LEQ = r'$\leq$'

# ---------------- S15 controlled stance test (values from the task brief / ASSESSMENT.md)
s15 = [('Canada', 'mpnet', .689, .153, .847, .193, 23), ('Canada', 'openai', .739, .128, .872, .193, 23), ('Canada', 'nomic', .468, .055, .945, .107, 23),
       ('Scotland', 'mpnet', .655, .142, .858, .175, 28), ('Scotland', 'openai', .678, .124, .876, .164, 28), ('Scotland', 'nomic', .439, .051, .949, .085, 28)]
body = ''
for i, r in enumerate(s15):
    body += f'{r[0] if r[1]=="mpnet" else ""} & {EMB_TEX[r[1]]} & {f3(r[2])} & {f3(r[3])} & {f3(r[4])} & {f3(r[5])} & {r[6]} \\\\\n'
    if i == 2: body += '\\midrule\n'
T['S15'] = table(body, r"\textbf{Controlled stance test of the coverage measure.} Test cases were built from real respondent passages (sentences of at least five words; base cases of 8--80 words, selected in a fixed pseudo-random order, up to 100 per kind and topic) and the coverage score was recomputed after a known change to the text. \emph{Exact-passage deletion}: the target text containing the passage and an unrelated passage from another respondent is reduced to the unrelated passage. \emph{Synthetic stance reversal}: the passage is embedded as ``Position A endorses this statement: [passage]'' and compared with the version that says ``rejects''. \emph{Condition-marker deletion}: in passages with one marker (``provided that'', ``on condition that'', ``only if'' or ``as long as'') the text from the marker onwards is deleted. The 2,451 cases are 800 passage deletions, 800 stance reversals, 800 ``disagreement deletions'' and 51 condition-marker deletions (23 Canada, 28 Scotland); the disagreement deletions are mathematically identical to the stance reversals under symmetric cosine similarity, so they are not independent tests. Columns give the mean drop in cosine similarity caused by each change, the mean similarity that remains between the endorsing and the rejecting version (1 means the score cannot tell them apart), and, for the condition test, the number of available cases $N$. The score drops sharply when the passage is removed but barely when the stance is reversed, so coverage measures topical closeness, not agreement.",
    'tab:s15', 'llL{2.3cm}L{2.3cm}L{2.3cm}L{2.3cm}r', head='Corpus & Embedding & Exact-passage deletion, mean drop & Synthetic stance reversal, mean drop & Reversed-stance mean similarity & Condition-marker deletion, mean drop & Condition cases $N$', tight=True)

# ---------------- S16 questions and official texts
CAN_Q = {
 'Education': ['What skills are required for a modern, digital economy, and how can Canada best support their development and deployment in the workforce?',
               'How can we enhance AI literacy in Canada, including awareness of AI’s limitations and biases?',
               'What can Canada do to ensure equitable access to AI literacy across regions, demographics and socioeconomic groups?'],
 'Trust': ['How can Canada build public trust in AI technologies while addressing the risks they present? What are the most important things to do to build confidence?',
           'What frameworks, standards, regulations and norms are needed to ensure AI products in Canada are trustworthy and responsibly deployed?',
           'How can Canada proactively engage citizens and businesses to promote responsible AI use and trust in its governance? Who is best placed to lead which efforts that fuel trust?']}
SCOT = {  # key: (label, official section(s), sentences, words, retained N)
 'q4': ('Q4', 'Grouping of procedures (Groups 1, 2, 3 and wider comments)', 76, 2135, '1,021'),
 'q7': ('Q7', 'Practitioner licenses (Questions 5--7)', 8, 216, '674'),
 'q20': ('Q20', 'Protected characteristics', 7, 180, '575'),
 'q21': ('Q21', 'Financial situation; Impact of regulation on businesses', 25, 608, '756'),
 'q22': ('Q22', 'Island communities', 5, 139, '549'),
 'q23': ('Q23', 'Child rights and wellbeing', 8, 254, '594')}
body = ''
for k, (nm, sec, ns, nw, N) in {'Education': ('Education', 'ISED report, p. 8 (six-sentence paragraph)', 6, 104, '2,496'), 'Trust': ('Trust', 'ISED report, p. 8 (six-sentence paragraph)', 6, 89, '2,757')}.items():
    qs = '\\newline '.join(f'({i+1}) {esc(q)}' for i, q in enumerate(CAN_Q[k]))
    body += f'Canada, {nm} & {qs} & {sec} & {ns} / {nw} & {N} \\\\\n\\midrule\n'
qjson_missing = False
for k, (nm, sec, ns, nw, N) in SCOT.items():
    f = inp('scotland', 'datasets', k, 'questions.json')
    if f:
        qd = json.load(open(f, encoding='utf8'))
        qs = esc(qd['question'])
    else:
        qs = r'[question wording not available: scotland/datasets/%s/questions.json]' % k
        qjson_missing = True
    body += f'Scotland, {nm} & {qs} & {sec} & {ns} / {nw} & {N} \\\\\n' + ('\\midrule\n' if k != 'q23' else '')
if qjson_missing: MISSING.append('S16: Scotland question wording (scotland/datasets/<q>/questions.json) not available; placeholders inserted')
T['S16'] = longtab(body, r'\textbf{Questions analyzed and the official texts paired with them.} Each row is one analysis unit: the consultation question(s) put to respondents, the official summary text that responses are compared with, the size of that text (sentences / words), and the number of respondents retained for analysis (those with a usable answer after cleaning). For Canada, the official text is the six-sentence paragraph on each theme in the report by Innovation, Science and Economic Development Canada (ISED), page 8; the paragraph is compared with the responses to all three questions of the theme. For Scotland, the official text is the section(s) of the consultation analysis report named in the third column.',
    'tab:s16', 'L{1.8cm}L{6.0cm}L{3.2cm}rr', 'Unit & Question wording & Official text paired with it & Sentences / words & Retained $N$')

# ---------------- S18 length and budget compliance
body = ''; prev = None
for r in budget_rows:
    t, m, runs, bud, sent, words, comp, att = r
    w = re.match(r'(\d+) \(range (\d+)–(\d+)\)', words)
    bs = re.match(r'(\d+) sent / ≤(\d+) words', bud)
    if prev and prev != t: body += '\\midrule\n'
    body += f'{(TOPIC_NAME[t] + (" (attempted; excluded)" if t == "q21" else "")) if prev != t else ""} & {MODEL_TEX[m]} & {bs.group(1)} sentences, {LEQ}{bs.group(2)} words & {sent} & {w.group(1)} ({w.group(2)}--{w.group(3)}) & {comp} \\\\\n'
    prev = t
T['S18'] = table(body, r'\textbf{Length and budget compliance of the generated summaries.} For each topic, summaries were generated with three language models: GPT-4o-mini (through the OpenAI API) and Qwen3.6-35B-A3B and GLM-4.7-flash (both run locally through Ollama). All runs used temperature 0.7 and three random seeds (1729, 1730 and 1731). Long inputs were summarized by map-reduce: the responses were split into portions of about 20,000 tokens, each portion was summarized, and the partial summaries were merged into one final summary. The final summary had to meet the official summary\'s budget, which is the same number of sentences and no more words than the official text. If a draft broke the budget, the model was asked again up to three times in a chain, and then up to three more times to condense the text. Columns: the budget; the mean number of sentences and the mean number of words over the three seeds (word range in brackets); and the number of runs (out of three) that met both limits. All Canadian and Q7 summaries met both budgets (3 of 3 runs in every case). Scotland Q21 was attempted but is excluded from the controlled comparison because 7 of 9 summaries missed the sentence budget; it has the largest budget, and the longest summary exceeded the word budget by 32\% (802 words against 608).',
    'tab:s18', 'llcrrc', head='Topic & Model & Budget & Mean sentences & Mean words (range) & Compliant runs', wide=True)

# ---------------- S19 agreement between summarizers
body = ''
for t in TOPICS:
    for emb in ['openai', 'mpnet']:
        d = LLM[(t, emb)]
        same = 'Yes' if len(set(d['lowest'])) == 1 else 'No'
        off = d['cov']['official'][2]
        for i, m in enumerate(MODELS):
            c = d['cov'][m][2].replace('±', r'$\pm$')
            sp = float(d['spear']['official'][m]); jr = d['jac'][f'official vs {m}']
            if i == 0:
                topic_label = TOPIC_NAME[t]
                if t == 'q21':
                    topic_label = r'\shortstack[l]{Scotland Q21\\(attempted;\\excluded)}'
                lead = f'\\multirow{{3}}{{*}}{{{topic_label}}} & \\multirow{{3}}{{*}}{{{EMB_TEX[emb]}}} & \\multirow{{3}}{{*}}{{{off}}} '
                kw = f'\\multirow{{3}}{{*}}{{{float(d["W"][0]):.3f}}} & \\multirow{{3}}{{*}}{{{same}}}'
            else:
                lead = '& & '; kw = '& '
            body += f'{lead}& {MODEL_TEX[m]} & {c} & {f3(sp)} & {float(jr[1]):.3f} ({float(jr[2]):.3f}) & {kw} \\\\\n'
        body += '\\midrule\n' if not (t == TOPICS[-1] and emb == 'mpnet') else ''
T['S19'] = longtab(body, r'\textbf{Agreement between the official summary and the generated summaries.} For each topic and embedding model, the table compares how well participants are covered by the official summary and by each generated summary (each model\'s coverage is first averaged over its three seeds; the Spearman correlations and Jaccard indices are computed on these seed-averaged values). ``Mean coverage\'\' is the average, over participants, of the largest cosine similarity between a response and any sentence of the summary. ``Spearman\'\' is the rank correlation of participant-level coverage between the official and the generated summary (1 means the same participants are covered best and worst). ``Jaccard\'\' is the overlap between the sets of the 10\% least-covered participants (1 means identical sets); the value in brackets is the overlap expected for random sets of the same size. ``Kendall\'s $W$\'\' measures how consistently the official and the three generated summaries rank the semantic regions of the responses by mean coverage (1 means identical rankings). The last column states whether all four summaries find the same region to be the worst covered. High agreement shows that the pattern of low coverage is shared across the tested summarizers under this protocol; it does not separate the contributions of the submissions, the prompt and the summarizer. For individual summaries, correlations with the official summary ranged from 0.61 to 0.93 (text-embedding-3-large) and Jaccard indices from 0.30 to 0.69 with the same embedding model for the Canadian topics and Q7. With all-mpnet-base-v2, one GLM Trust run had correlation 0.025 and Jaccard 0.083. Scotland Q21 was attempted but is excluded from the controlled comparison because 7 of 9 summaries missed the sentence budget and one exceeded the word budget by 32\% (Table~\ref{tab:s18}); its rows are shown for completeness.',
    'tab:s19', 'L{1.8cm}L{1.3cm}L{1.2cm}L{2.0cm}L{2.0cm}L{1.2cm}L{1.7cm}L{1.2cm}L{1.2cm}', head='Topic & Embed. model & Official mean coverage & Summarizer & Mean coverage (generated) & Spearman $\\rho$ & Jaccard, bottom decile (random mean) & Kendall $W$ & Same worst region?', small='\\scriptsize')

# ---------------- S20 coverage by closed-answer group (Scotland)
RA = pd.read_csv(inp('stance_groups', 'results_all.csv'))
def grp(r): return f'{r.minority_label} vs {r.majority_label}'
def ci(a, b, c): return f'{sg(a)} [{sg(b)}, {sg(c)}]'
def srow(r, show_q=True, show_e=True, qn=None):
    return (f'{qn or ""} & {EMB_TEX[r.embedding_model]} & {r.n_min}/{r.n_maj} & {ci(r["diff"], r.ci_lo, r.ci_hi)} & {fmtp(r.p_perm)} & {fmtp(r.p_holm)} & '
            f'{ci(r.adj_coef, r.adj_lo, r.adj_hi)} & {fmtp(r.adj_p_holm) if "adj_p_holm" in r else "--"} & {100*r.share_bottom10_min:.1f} / {100*r.share_bottom10_maj:.1f} \\\\\n')
head_ab = 'Question & Embedding & $n$ (first/second group) & Mean difference [95\\% CI] & Permutation $p$ & Holm $p$ & Length-adjusted difference [95\\% CI] & Holm $p$ (adjusted) & Bottom-decile share, \\% (first / second) '
off = RA[RA.source == 'official']
body = '\\multicolumn{9}{l}{\\textit{Panel A. Official summary, closed-answer groups (first group = minority, second = majority)}}\\\\\n'
for q, aid, lab in [('q4', 'q4_level_vs_proposal', 'Q4: oppose vs support'), ('q7', 'q7_Q5', 'Q7 (Q5): oppose vs support'), ('q23', 'q23_Q19', 'Q23 (Q19): oppose vs support')]:
    for emb in ['openai', 'mpnet', 'nomic']:
        r = off[(off.analysis_id == aid) & (off.embedding_model == emb)].iloc[0]
        assert f'{r.minority_label} vs {r.majority_label}' == 'oppose vs support'
        body += srow(r, qn=lab if emb == 'openai' else '')
    body += '\\midrule\n' if q != 'q23' else ''
body += '\\midrule\n\\multicolumn{9}{l}{\\textit{Panel B. Official summary, organization vs individual respondents (OpenAI embedding)}}\\\\\n'
for q in ['q4', 'q7', 'q20', 'q21', 'q22', 'q23']:
    r = off[(off.analysis_id == f'{q}_resp_type') & (off.embedding_model == 'openai')].iloc[0]
    body += srow(r, qn=q.upper())
body += '\\midrule\n\\multicolumn{9}{l}{\\textit{Panel C. Generated summaries, Q7 (Q5): oppose vs support (seed-averaged coverage)}}\\\\\n'
gen = RA[(RA.source == 'generated') & (RA.analysis_id == 'q7_Q5')]
for m in MODELS:
    for emb in ['openai', 'mpnet']:
        r = gen[(gen.summarizer == m) & (gen.embedding_model == emb)].iloc[0]
        body += srow(r, qn=MODEL_TEX[m] if emb == 'openai' else '')
T['S20'] = table(body, r'\textbf{Coverage of Scottish respondents by their own closed answers.} Coverage is the similarity between a respondent\'s free-text answer and the closest sentence of the summary. The table compares mean coverage between two groups of respondents. Panel A uses the closed-answer position on the proposal (oppose or support; neutral respondents excluded) for the three pre-specified questions; Panel B compares organizations with individuals for all six questions; Panel C repeats the Q7 analysis for the generated summaries. ``Mean difference\'\' is first group minus second group (negative: the first group is covered less), with a 95\% confidence interval. ``Permutation $p$\'\' tests whether the difference could arise by chance; ``Holm $p$\'\' corrects it for the number of tests in the family. The length-adjusted difference is the same comparison after accounting for how long each response is; its Holm $p$ is given next to it. The last column gives the share of each group that falls in the lowest-covered 10\% of all respondents (10\% would mean no difference).  Small $p$-values are written in powers of ten.',
    'tab:s20', 'L{1.9cm}L{1.3cm}L{1.3cm}L{2.3cm}L{1.4cm}L{1.2cm}L{2.3cm}L{1.2cm}L{1.6cm}', head=head_ab, small='\\scriptsize', tight=True)

# ---------------- S21 idea audit
VT = pd.read_csv(inp('idea_audit', 'validation_table.csv'))
SS = pd.read_csv(inp('idea_audit', 'summary_shares.csv'))
SE = pd.read_csv(inp('idea_audit', 'sensitivity.csv'))
NC = 8
body = f'\\multicolumn{{{NC}}}{{L{{13.4cm}}}}{{\\textit{{Panel A. Synthetic interventions (k of n cases; 95\\% Wilson interval)}}}}\\\\\n'
body += 'Topic & Test & Rule or judge & $k/n$ & Rate (\\%) & 95\\% CI & & \\\\\n\\midrule\n'
prev = None
for _, r in VT.iterrows():
    test = r['test'].split(':')[0]
    rule = r['judge/rule']
    rule = esc(rule).replace('cos\\textgreater{}', 'cos').replace('>=', r'$\geq$')
    rule = rule.replace('entail<0.5', r'entail$<$0.5').replace('entail>=0.5', r'entail$\geq$0.5').replace('cos>=tau', r'cos$\geq\tau$').replace('tau = 10th', r'$\tau$ = 10th').replace('->', r'$\rightarrow$')
    rule = rule.replace('(cos$\\geq\\tau$=', '(cos$\\geq\\tau$=')
    if prev and prev != r.topic: body += '\\midrule\n'
    prev = r.topic
    body += f'{TOPIC_NAME[r.topic]} & {esc(test)} & {rule} & {r.k}/{r.n} & {f1(r["rate %"])} & {esc(r["95% Wilson CI"])} & & \\\\\n'
body += f'\\midrule\n\\multicolumn{{{NC}}}{{L{{13.4cm}}}}{{\\textit{{Panel B. Share of recurring ideas (at least 10 respondents) by label; generated summaries averaged over the three seeds (\\%)}}}}\\\\\n'
body += 'Topic & Summary & Ideas & Missed & Covered & Stance mismatch & Uncertain & Relocated \\\\\n\\midrule\n'
for t in TOPICS:
    d = SS[SS.topic == t]
    rows = [('Official', d[d.model == 'official'].iloc[0])] + [(MODEL_TEX[m], d[d.model == m][['%MISSED', '%COVERED', '%STANCE_MISMATCH', '%UNCERTAIN', '%RELOCATED', 'n_ideas']].mean()) for m in MODELS]
    for i, (nm, r) in enumerate(rows):
        body += f'{TOPIC_NAME[t] if i == 0 else ""} & {nm} & {int(r.n_ideas)} & {f1(r["%MISSED"])} & {f1(r["%COVERED"])} & {f1(r["%STANCE_MISMATCH"])} & {f1(r["%UNCERTAIN"])} & {f1(r["%RELOCATED"])} \\\\\n'
    body += '\\midrule\n' if t != TOPICS[-1] else ''
body += f'\\midrule\n\\multicolumn{{{NC}}}{{L{{13.4cm}}}}{{\\textit{{Panel C. Official summary: sensitivity to the minimum number of respondents per idea (clustering threshold 0.3; \\%)}}}}\\\\\n'
body += 'Topic & Min.\\ respondents & Ideas & Missed & Covered & Stance mismatch & Uncertain & Relocated \\\\\n\\midrule\n'
for t in TOPICS:
    d = SE[SE.topic == t]
    for i, (_, r) in enumerate(d.iterrows()):
        body += f'{TOPIC_NAME[t] if i == 0 else ""} & {r.min_resp} & {r.n_ideas} & {f1(r["%MISSED"])} & {f1(r["%COVERED"])} & {f1(r["%STANCE_MISMATCH"])} & {f1(r["%UNCERTAIN"])} & {f1(r["%RELOCATED"])} \\\\\n'
    body += '\\midrule\n' if t != TOPICS[-1] else ''
body += f'\\midrule\n\\multicolumn{{{NC}}}{{L{{13.4cm}}}}{{\\textit{{Panel D. Are ideas of opposition or concern missed more often? Share missed (\\%) and adjusted odds ratio (OR)}}}}\\\\\n'
body += 'Min.\\ respondents & Summaries & Scope & Idea--summary pairs & Oppose/ concern missed (\\%, $n$) & Other missed (\\%, $n$) & Adjusted OR [95\\% CI] & Adjusted $p$ \\\\\n\\midrule\n'
for s_ in STANCE:
    pairs = f'{int(s_["pairs"]):,}'
    opp = f'{s_["pct_missed_opp"]:.1f} ({int(s_["k_opp"]):,}/{int(s_["n_opp"]):,})'; oth = f'{s_["pct_missed_other"]:.1f} ({int(s_["k_oth"]):,}/{int(s_["n_oth"]):,})'
    if s_['status'] == 'ok':
        tail = f'{s_["adj_or"]:.2f} [{s_["adj_lo"]:.2f}, {s_["adj_hi"]:.2f}] & {fmtp(s_["adj_p"])}'
    else:
        tail = '\\multicolumn{2}{L{3.0cm}}{not estimable (no convergence/separation: no Scottish idea labeled missed)}'
    body += (f'{int(s_["min_resp"])} & {s_["summaries"].capitalize()} & {"Canada" if s_["scope"] == "Canada" else "All four topics"} & {pairs} & {opp} & {oth} & {tail} \\\\\n')

# Panel E: synthetic interventions by stance (Canadian topics), computed from validation_rows_t0.30.parquet + ideas_t0.30.parquet
_vr = []
for _t in ['education', 'trust']:
    _wd = inp('idea_audit', 'work', _t)
    _v = pd.read_parquet(os.path.join(_wd, 'validation_rows_t0.30.parquet'))
    _i = pd.read_parquet(os.path.join(_wd, 'ideas_t0.30.parquet'))[['idea_id', 'stance']]
    _vr.append(_v.merge(_i, on='idea_id', how='left'))
VR = pd.concat(_vr, ignore_index=True)
assert VR.stance.notna().all()
VR['grp'] = np.where(VR.stance.isin(['oppose', 'concern']), 'Opposition/concern', 'Other')
def _cnt(test, label, grp):
    d = VR[(VR.test == test) & (VR.grp == grp)]
    return int((d.label == label).sum()), len(d)
SYN = []
for t_, test, label in [('Planted inclusion labeled COVERED', 'inclusion', 'COVERED'), ('Planted inclusion labeled MISSED', 'inclusion', 'MISSED'),
                        ('Unrelated-summary negative control labeled MISSED', 'negctrl', 'MISSED')]:
    for g_ in ['Opposition/concern', 'Other']:
        SYN.append((t_, g_) + _cnt(test, label, g_))
body += f'\\midrule\n\\multicolumn{{{NC}}}{{L{{13.4cm}}}}{{\\textit{{Panel E. Synthetic interventions by stance of the idea (Canadian topics; 95\\% Wilson interval)}}}}\\\\\n'
body += 'Intervention & & Idea stance & $k/n$ & Rate (\\%) & 95\\% CI & & \\\\\n\\midrule\n'
for t_, g_, k_, n_ in SYN:
    lo_, hi_ = wilson(k_, n_)
    body += f'\\multicolumn{{2}}{{L{{4.2cm}}}}{{{esc(t_) if g_.startswith("Opp") else ""}}} & {g_} & {k_}/{n_} & {100*k_/n_:.1f} & [{100*lo_:.1f}, {100*hi_:.1f}] & & \\\\\n'
_d = {(a_, b_): (k_, n_) for a_, b_, k_, n_ in SYN}
_f = _d[('Planted inclusion labeled MISSED', 'Opposition/concern')], _d[('Planted inclusion labeled MISSED', 'Other')]
fm = {'opp': _f[0][0] / _f[0][1], 'oth': _f[1][0] / _f[1][1]}
_n = _d[('Unrelated-summary negative control labeled MISSED', 'Opposition/concern')], _d[('Unrelated-summary negative control labeled MISSED', 'Other')]
nc = {'opp': _n[0][0] / _n[0][1], 'oth': _n[1][0] / _n[1][1]}
_o = stance_row(10, 'official', 'Canada')
obs = {'opp': _o['k_opp'] / _o['n_opp'], 'oth': _o['k_oth'] / _o['n_oth']}
cor = {g: (obs[g] - fm[g]) / (nc[g] - fm[g]) for g in obs}
body += (f'\\multicolumn{{{NC}}}{{L{{13.4cm}}}}{{Adjusted missed share, official summaries (Canada, at least 10 respondents): opposition/concern {cor["opp"]:.2f} '
         f'(observed {obs["opp"]:.2f}), other {cor["oth"]:.2f} (observed {obs["oth"]:.2f}). Illustrative calculation assuming that these synthetic response rates apply to the actual summaries; this assumption has not been validated.}}\\\\\n')
jt = open(inp('idea_audit', 'RESULTS.md'), encoding='utf8').read()
jt = jt[jt.index('## Judge agreement'):]
kap = {t: float(re.search(rf'\| {t} \| official \| \d+ \| ([-\d.]+) \|', jt).group(1)) for t in ['education', 'trust']}
body += f'\\multicolumn{{{NC}}}{{L{{13.4cm}}}}{{Cohen\'s kappa between the two judges (official summaries; binary covered vs. not covered): Education {kap["education"]:.2f}, Trust {kap["trust"]:.2f}.}}\\\\\n'
print('Panel E adjusted shares', cor, SYN)
assert (_d[('Planted inclusion labeled COVERED', 'Opposition/concern')], _d[('Planted inclusion labeled COVERED', 'Other')]) == ((19, 28), (38, 40))
assert abs(cor['opp']-0.66) < 0.006 and abs(cor['oth']-0.29) < 0.006, cor
head21 = 'Topic & Test / summary & Rule / ideas & Result & & & & '
T['S21'] = longtab(body, r'\textbf{Synthetic interventions, sensitivity and statistical test for the idea audit.} The audit extracts the ideas that recur among respondents and asks, for each summary, whether the idea is present. An idea is labeled MISSED when both judges (a natural-language-inference model and a language-model judge) find it absent, COVERED when both find it present, STANCE\_MISMATCH when the summary discusses the topic with the opposite stance, RELOCATED when it appears in a different part of the text, and UNCERTAIN when the judges disagree. \textbf{Panel A} checks the method with synthetic interventions: planted deletions (the sentences that the inference model judged to express the idea are removed; the remaining text may still express it, so a missed label is expected but not certain), planted paraphrases, called inclusions (should be found present), planted opposite-stance versions (should not count as covered) and an unrelated summary (everything should be missed); each row gives the share of synthetic interventions that behave as expected under one rule, with its 95\% Wilson interval. \textbf{Panel B} gives the label shares for the official summary and, for each model, the mean over the three seeds. \textbf{Panel C} shows how the official-summary shares change with the minimum number of respondents required for an idea to be counted. \textbf{Panel D} compares the share of MISSED ideas between ideas expressing opposition or concern and all other ideas, for ideas with at least 5 or at least 10 respondents. The adjusted odds ratio comes from a logistic regression that also accounts for the number of respondents per idea and the topic, with standard errors clustered by idea. A ratio above 1 means opposition or concern ideas are missed more often. \textbf{Panel E} repeats two of the synthetic interventions separately for ideas of opposition or concern and for all other ideas. The first line below it gives an adjusted missed share for the official summaries, computed as (observed missed share $-$ rate of planted inclusions labeled MISSED) / (negative-control MISSED rate $-$ rate of planted inclusions labeled MISSED); it is an illustrative calculation that assumes that these synthetic response rates apply to the actual summaries, an assumption that has not been validated, and it rests on few synthetic cases. The second line gives the agreement (Cohen\'s kappa) between the two judges. In Panel D the logistic model for official summaries with at least 10 respondents over all four topics could not be estimated (no Scottish idea is labeled missed, which separates the outcome perfectly). In the all-topic rows for generated summaries, the Q21 generated summaries are excluded because 7 of 9 missed their length budget; Q21 generated summaries in Panel B are shown for completeness only and are not part of the controlled comparison. For generated summaries the same idea appears once per summary, so Fisher exact tests that treat idea--summary pairs as independent would overstate certainty and are not reported; the clustered model is the basis for inference.',
    'tab:s21', 'L{1.7cm}L{2.3cm}L{2.6cm}L{1.3cm}L{1.4cm}L{1.5cm}L{1.5cm}L{1.5cm}', None, small='\\scriptsize', ncol=8)

# ---------------- S22 ten most prevalent ideas missed by the official summary
it = open(inp('idea_audit', 'RESULTS.md'), encoding='utf8').read()
def missed_tbl(topic):
    m = re.search(rf'\n## {topic}\n.*?### 10 most prevalent ideas MISSED by the official summary \(for the paper\)\n+(\|.*?)\n\n', it, re.S)
    rows = [l for l in m.group(1).split('\n') if l.startswith('|')][2:]
    out = []
    for l in rows:
        c = [x.strip() for x in l.strip('|').split(' | ')]
        out.append(c)
    return out
body = ''
for t in ['education', 'trust']:
    body += f'\\multicolumn{{6}}{{l}}{{\\textit{{{TOPIC_NAME[t]}}}}}\\\\\n'
    for i, c in enumerate(missed_tbl(t)):
        stmt, stance, n, pct, frac, q = c[0], c[1], c[2], c[3], c[4], c[5]
        k = int(round(float(frac) * 10))
        qq = q.strip().strip('"').replace('"', "'") if len(q.split()) <= 15 else '--'
        qq = f'``{esc(qq)}\'\'' if qq != '--' else '--'
        body += f'{i+1} & {esc(stmt)} & {stance} & {n} ({pct}\\%) & {k} of 10 & {qq} \\\\\n'
    body += '\\midrule\n' if t == 'education' else ''
T['S22'] = longtab(body, r'\textbf{The ten most prevalent ideas missed by the official summary, for Education and Trust.} Each row is a canonical statement for an idea cluster containing at least 10 respondents (claims extracted from the responses were grouped by similarity and stance, and the statement was written by a language model). ``Stance\'\' is the position taken by the idea (oppose, concern, support or suggestion). ``Respondents\'\' is the number of distinct respondents whose claims are in the cluster, with their share of all respondents for the topic; these respondents need not have expressed every detail of the statement. The official summary is listed as missing the idea because the two automated judges (a natural-language-inference model and a language-model judge) both labeled it MISSED. ``Missed in\'\' counts how many of the ten summaries (the official one and nine generated ones: three models times three seeds) received the MISSED label; a high count means that the omission is shared by many summarizers. The last column shows one short illustrative quotation from a respondent (shown only when it is at most 15 words long).',
    'tab:s22', 'rL{5.2cm}L{1.5cm}L{2.0cm}L{1.2cm}L{3.4cm}', 'Rank & Canonical statement & Stance & Respondents (share) & Missed in & Example quotation')

# ---------------- S17 Scotland full results
SC = pd.read_csv(inp('scotland_analysis', 'scotland_analysis_summary.csv'))
EMAP = {'text-embedding-3-large': 'openai', 'all-mpnet-base-v2': 'mpnet', 'nomic-embed-text (robustness)': 'nomic'}
SC['emb'] = SC.embedding_model.map(EMAP)
QS = ['q4', 'q7', 'q20', 'q21', 'q22', 'q23']
def ptxt(v): return '--' if pd.isna(v) else fmtp(v)
bodyA = ''; bodyB = ''
for q in QS:
    for emb in ['openai', 'mpnet', 'nomic']:
        r = SC[(SC.topic == q) & (SC.emb == emb)].iloc[0]
        bodyA += (f'{q.upper() if emb == "openai" else ""} & {EMB_TEX[emb]} & {int(r.n_retained)} & {int(r.n_regions)} & {f3(r.official_mean_coverage)} & {f3(r.bottom_decile_mean)} & '
                  f'{f3(r.low_coverage_rate)} & {f3(r.gini)} & {f3(r.exact_random_null_mean)} & {ptxt(r.exact_random_mean_p)} & {f3(r.exact_random_null_bottom_decile)} & {ptxt(r.exact_random_bottom_decile_p)} \\\\\n'.replace('--- ', '--'))
        if emb != 'nomic':
            bodyB += (f'{q.upper() if emb == "openai" else ""} & {EMB_TEX[emb]} & {sg(r.mean_opt_gain)} [{sg(r.mean_opt_gain_ci_low)}, {sg(r.mean_opt_gain_ci_high)}] & {fmtp(r.mean_opt_p_holm)} & '
                      f'{sg(r.tail_opt_gain)} [{sg(r.tail_opt_gain_ci_low)}, {sg(r.tail_opt_gain_ci_high)}] & {fmtp(r.tail_opt_p_holm)} \\\\\n')
    bodyA += '\\midrule\n' if q != 'q23' else ''
    bodyB += '\\midrule\n' if q != 'q23' else ''
bodyA = bodyA.replace('nan', '--')
tabA = ('\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}{L{1.3cm}L{1.3cm}L{0.8cm}L{0.9cm}L{1.1cm}L{1.2cm}L{1.2cm}L{0.9cm}L{1.3cm}L{0.9cm}L{1.4cm}L{1.3cm}}\n\\toprule\nQuestion & Embedding & $N$ & Regions & Mean coverage & Bottom-decile mean & Low-coverage rate & Gini & Random-window mean & $p$ (mean) & Random-window bottom decile & $p$ (bottom decile) \\\\\n\\midrule\n' + bodyA + '\\bottomrule\n\\end{tabular}')
tabB = ('\\begin{tabular}{llcccc}\n\\toprule\nQuestion & Embedding & Mean-optimized gain [95\\% CI] & Holm $p$ & Tail-oriented gain [95\\% CI] & Holm $p$ \\\\\n\\midrule\n' + bodyB + '\\bottomrule\n\\end{tabular}')
T['S17'] = ('\\begin{table}[H]\n\\revon\n\\centering\\small\n\\caption{\\textbf{Scottish consultation: coverage of the official text and held-out improvement, by question and embedding model.} '
  'Coverage is the highest cosine similarity between a respondent\'s answer and any sentence of the official text. '
  '\\textbf{Panel A:} $N$ is the number of retained respondents; regions are the clusters used for the regional analyses; mean coverage, bottom-decile mean (average of the lowest-covered 10\\% of respondents), low-coverage rate (share of respondents more than one standard deviation below the mean) and Gini coefficient (inequality of coverage) describe the official text. '
  'The random benchmark averages coverage over 1,000 pseudo-summaries, each containing one random window per official sentence, matched to its word count. The one-sided $p$-value is the number of pseudo-summaries that match or exceed the official score plus one, divided by 1,001. '
  '\\textbf{Panel B:} gain is the difference between a summary selected on training respondents and evaluated on held-out respondents (5$\\times$5-fold cross-fitting, same length budget) and the official text; positive values mean that the official text could have covered held-out respondents better. The mean-optimized selection targets mean coverage and the tail-oriented selection targets the bottom-decile mean. Intervals are 95\\% participant-bootstrap intervals; Holm $p$ corrects the one-sided test across the 24 primary comparisons. Nomic is a robustness model and has coverage statistics only (dashes).}\n'
  '\\label{tab:s17}\n{' + tabA + '}\n\\vspace{1.2em}\n\n' + tabB + '\n\\end{table}\n')

# ---------------- S23 distribution distances
FD = open(inp('metric_sensitivity', 'FINDINGS.md'), encoding='utf8').read()
part1, part2 = FD.split('## Frozen-group distances')
grp = {}
for m in re.finditer(r'\| (\S+) / (\S+) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \|', part2):
    grp[(m.group(1), m.group(2), int(m.group(3)))] = (m.group(5), m.group(6))
DN = {'canada_education': 'Canada Education', 'canada_trust': 'Canada Trust'}
for q in QS: DN[f'scotland_analysis_{q}'] = f'Scotland {q.upper()}'
body = ''; prevd = None
rows = re.findall(r'\| (\S+) / (\S+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (\S+) \| (\S+) \| (\d+/\d+) / (\d+/\d+) \|', part1)
assert len(rows) == 66, len(rows)
def dash(x): return '--' if x in ('—', '-') else x
for ds, be, dim, w1, w2, mmd, r1, r2, c1, c2 in rows:
    be2 = 'nomic' if be.startswith('nomic') else be
    if prevd and prevd != ds: body += '\\midrule\n'
    first = prevd != ds; prevd = ds
    nomic = be2 == 'nomic'
    g = grp.get((ds, be2, int(dim)))
    body += (f'{DN[ds] if (first) else ""} & {EMB_TEX[be2]} & {dim} & {w1} & {w2} & {mmd} & {dash(r1)} & {dash(r2)} & '
             f'{"--" if nomic else c1.replace("/", " of ")} & {"--" if nomic else c2.replace("/", " of ")} & {g[0] if g else "--"} & {g[1] if g else "--"} \\\\\n')
T['S23'] = longtab(body, r'\textbf{Distances between the distribution of respondents and the official summary.} As an alternative to coverage, this table reports how far the official text lies from the cloud of respondent answers in embedding space, after reducing the embeddings to 20, 50 or 100 principal components fitted on respondents only. W1 and W2 are Wasserstein (optimal-transport) distances and MMD$^2$ is the squared maximum mean discrepancy with a median-distance bandwidth; smaller values mean a closer match. ``Rank $\rho$\'\' columns are Spearman correlations, over the official text and 20 random text windows of the same length, between the ranking by W1 and by W2 and between the ranking by MMD$^2$ and by W2 (values near 1 mean the metrics order the texts alike). ``Official closer than random\'\' counts the random windows (out of 20) that the official text beats on W2 and on MMD$^2$; nomic has no random references (dashes). The last two columns, available only for the Canadian datasets, are rank correlations between the metrics when ranking the respondent groups (regions) by their distance to the official text. Distances are only comparable within one dataset, embedding model and dimension. Unlike Table~\ref{tab:transport}, which fits PCA jointly on participant and official-summary embeddings, this comparison fits PCA on participants alone so that all targets use the same axes; this accounts for the different Canadian W2 values.',
    'tab:s23', 'L{1.7cm}L{1.4cm}L{0.8cm}L{0.9cm}L{0.9cm}L{1.0cm}L{0.9cm}L{1.0cm}L{1.0cm}L{1.0cm}L{1.0cm}L{1.0cm}', 'Dataset & Embedding & PCA dims & Official W1 & Official W2 & Official MMD$^2$ & W1--W2 rank $\\rho$ & MMD$^2$--W2 rank $\\rho$ & Closer than random, W2 & Closer than random, MMD$^2$ & Group rank $\\rho$, W1--W2 & Group rank $\\rho$, MMD$^2$--W2', small='\\scriptsize')

# Keys/TeX labels are stable identifiers for the table fragments.
# Printed numbers follow first citation in main.tex; insert by label.
printed_numbers = {'S16': 1, 'S17': 14, 'S18': 15, 'S19': 16, 'S15': 17, 'S20': 18, 'S22': 19, 'S21': 20, 'S23': 23}
order = list(printed_numbers)

# Keep oversized Scottish panels on separate pages with the same table number.
T['S17'] = T['S17'].replace('\\vspace{1.2em}\n\n', '\\end{table}\n\\begin{table}[H]\n\\revon\\centering\\small\n\\ContinuedFloat\n\\caption{\\textbf{Scottish consultation (continued): Panel B, held-out benchmark gains.} Intervals are conditional 95\\% participant-bootstrap intervals; Holm adjustment covers all 24 primary comparisons.}\n')
T['S17'] = T['S17'].replace(tabB, '\\resizebox{\\linewidth}{!}{' + tabB + '}')
T['S15'] = T['S15'].replace('\\begin{tabular}', '\\resizebox{\\linewidth}{!}{\\begin{tabular}', 1).replace('\\end{tabular}', '\\end{tabular}}')
T['S21'] = T['S21'].replace('\\setlength{\\tabcolsep}{3pt}', '\\setlength{\\tabcolsep}{2pt}')

with open(os.path.join(OUT, 'supp_tables_new.tex'), 'w', encoding='utf8') as f:
    f.write('% Generated supplementary table fragments. Insert each fragment by its stable label.\n\n')
    for k in order:
        f.write(f'% ---------------- Table S{printed_numbers[k]} (stable key {k})\n' + '\\setcounter{table}{' + str(printed_numbers[k] - 1) + '}\n' + T[k] + '\n')
print('tables written; missing:', MISSING)
