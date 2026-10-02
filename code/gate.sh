#!/usr/bin/env bash
# Submission gate (writing standard 12.21): every check exits non-zero on failure.
set -e
cd "$(dirname "$0")/.."
MAIN=TMC-2026-07-3202-main
latexmk -silent $MAIN.tex >/dev/null
pages=$(pdfinfo out/$MAIN.pdf | awk '/^Pages/{print $2}')
[ "$pages" -eq 18 ] || { echo "FAIL: $pages pages (limit 18)"; exit 1; }
! grep -q '^!' out/$MAIN.log || { echo "FAIL: LaTeX errors"; exit 1; }
! grep -q 'Overfull' out/$MAIN.log || { echo "FAIL: overfull boxes"; exit 1; }
! grep -q 'undefined' out/$MAIN.log || { echo "FAIL: undefined references"; exit 1; }
# mangled control sequences and bare macro stubs (12.11)
! perl -ne 'print "$ARGV:$.: $_" if /[\x00-\x08\x0b\x0c\x0e-\x1f]/' $MAIN.tex refs.bib TMC_Response_Letter.tex supp_proof.tex fig1_system_r1.tex | grep -q . || { echo "FAIL: control character"; exit 1; }
! grep -nE '(^|[^\\a-zA-Z])(imes|ef\{|abel\{|extbf\{|rac\{|ootnote\{|ite\{|egin\{|nd\{)' $MAIN.tex | grep -v 'quadratically' || { echo "FAIL: bare macro stub"; exit 1; }
# configuration drift (12.10): stale literals from earlier builds
! grep -nE 'width=0\.79|2\.87|0\.82\\linewidth|main\.pdf' $MAIN.tex || { echo "FAIL: stale literal"; exit 1; }
# every bare 4% (the per-user pilot fraction) is qualified by "user" or by the 16% total within two lines
python3 - <<'PY'
import re,sys
L=open("TMC-2026-07-3202-main.tex",encoding="utf-8").read().split("\n")
bad=[i+1 for i,l in enumerate(L) if re.search(r"(^|[^0-9.])4\\%",l) and not any(("user" in x or "16\\%" in x) for x in L[max(0,i-1):i+2])]
if bad: print("FAIL: unqualified 4% at lines",bad); sys.exit(1)
PY
# bibliography: every entry cited, every cite present
python3 - <<'PY'
import re,sys
s=open("TMC-2026-07-3202-main.tex",encoding="utf-8").read(); b=open("refs.bib",encoding="utf-8").read()
c=set(k.strip() for m in re.finditer(r"\\cite\{([^}]*)\}",s) for k in m.group(1).split(","))
k=set(re.findall(r"@\w+\{([^,]+),",b))
if c!=k: print("FAIL bib:",sorted(c-k),sorted(k-c)); sys.exit(1)
PY
echo "gate passed: $pages pages, 0 errors, 0 overfull"
