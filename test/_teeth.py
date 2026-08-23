# -*- coding: utf-8 -*-
"""`_teeth.py` — СПІЛЬНИЙ ХАРНЕС АСЕРТІВ ІЗ ЗУБАМИ (слово автора «став», 2026-07-22).

ЩО ЦЕ ЛІКУЄ
-----------
Клас `S1055::T9` — **assert, що не може завалитись**. Твердження ІСТИННЕ, тест ЗЕЛЕНИЙ,
а свідчення порожнє, бо детектор однаково каже ТАК у світі, де об'єкт правильний, і в
світі, де він зламаний. Такий тест не є свідком: він не розрізняє світи.

Інстанси, на яких харнес побудовано (не гіпотетичні — обидва зловлені в репо):
  · `S1055::T9` — assert «фактор = 1/колон-кривина»: `(T/C)/T ≡ 1/C ∀C`, тотожність;
  · `S1059::T3` — «жоден член ряду не порожній»: істинне для БУДЬ-ЯКОГО невиродженого
    оператора, у т.ч. для того, де одиниця справді 2 хопи ⟹ нічого не розрізняє;
  · `S1059::T4` — «лічба на бонд не рухається під P»: `P` до обчислення не доходив,
    чотири ітерації рахували ту саму величину.

ЧОМУ САМЕ ТАКА СИГНАТУРА (а не «правило в README»)
---------------------------------------------------
Карбований урок S1056: **урок стає гейтом лише ставши ОБОВ'ЯЗКОВИМ ПОЛЕМ.** Тому тут
детектор — це **функція від СВІТУ**, а не готове булеве: готове булеве не можна
перепитати на іншому об'єкті, і саме тому локальні `def ok(cond, msg)` у test_1..test_3
структурно не здатні спіймати цей клас. Забути негконтроль неможливо не з дисципліни,
а тому, що `ok()` без нього не викликається.

МЕЖІ — ЧИТАТИ ПЕРЕД ТИМ, ЯК ПОКЛАСТИСЬ (інакше рядок бреше)
------------------------------------------------------------
1. Харнес міряє **ЧУТЛИВІСТЬ, не правильність.** Тест може мати зуби і бути про не те.
2. Він **не перевіряє, чи негативний світ чесний.** Можна назвати світ, де детектор
   падає тривіально, і купити фальшивий зуб. Халтура стає дорожчою, не неможливою.
3. Він **не ловить «сказано замість виведено»** — відсутній вивід не є порожнім тестом.
   Це інший шар (доказ / Lean), і цей файл його не заміняє й не конкурує з ним.
4. `ok_contrast` вимагає ПАРУ, бо справжня інваріантність і сліпота машинерії дають
   однаковий вивід. Без величини, що МУСИТЬ рухатись, «не зрушило» нічого не важить.

ВІДНОШЕННЯ ДО НАПРЯМКУ ОМЕГИ («джерело величин + асерт проти нього», 2026-07-22)
--------------------------------------------------------------------------------
Різні дефекти, не конкуренти: її механізм ловить **РОЗСИНХРОН** (число розійшлось із
джерелом), цей — **ПОРОЖНЕЧУ** (тест не розрізняє світи). 2026-07-22 репо несло обидва.

САМОПЕРЕВІРКА
-------------
`python _teeth.py` прогонить харнес на ЧОТИРЬОХ реальних детекторах S1059 і вимагає,
щоб два були позначені порожніми, а два пройшли. Харнес, який перевіряє зуби, мусить
мати власні: якщо він перестане ловити — його ж самотест падає з exit 1.
"""
import sys

_pass, _fail, _void = [], [], []


def _say(s=''):
    print(s)


def ok(detector, world, msg, *, must_fail_on):
    """Асерт, який зобов'язаний уміти впасти.

    detector    — ФУНКЦІЯ ВІД СВІТУ, що повертає bool. Не готове значення: готове
                  значення не можна перепитати на іншому об'єкті.
    world       — наш об'єкт (той, про який твердження).
    must_fail_on— (назва, світ) або список таких пар: об'єкт(и), де детектор МУСИТЬ
                  сказати НІ. Якщо він і там каже ТАК — assert самознищується (☠),
                  бо він нічого не розрізняє.

    ★Негконтроль тут прогоняє ТОЙ САМИЙ `detector`, що несе свідчення — це не угода,
    а єдине, що дозволяє сигнатура. Клас Альфи (S1059 ред.2): мало МАТИ негконтроль,
    він мусить бити в той самий детектор; інакше маємо два зелені, які нічого один
    про одного не знають (рівно S1059::T3: тест міряв ненульовість члена ряду,
    негконтроль — досяжність у r-просторі).

    Повертає вирок детектора на `world`; ☠ рахується окремо від ✗ і теж дає exit≠0.
    """
    negatives = must_fail_on if isinstance(must_fail_on, list) else [must_fail_on]
    if not negatives:
        raise ValueError("must_fail_on is empty: an assert without a negative world is forbidden")
    blind = [name for name, w in negatives if detector(w)]
    if blind:
        _void.append(msg)
        _say(f"  ☠ EMPTY  {msg}")
        for name in blind:
            _say(f"       the detector said YES on «{name}» too ⟹ it does NOT distinguish worlds")
        return False
    verdict = bool(detector(world))
    (_pass if verdict else _fail).append(msg)
    _say(f"  {'✓' if verdict else '✗ FAIL'} {msg}")
    _say(f"       (tooth: said NO on «{negatives[0][0]}» ⟹ able to fail)")
    return verdict


def ok_contrast(probe, params, msg):
    """Асерт про ІНВАРІАНТНІСТЬ під параметром — з обов'язковим контрастом.

    probe(p) → (що_стоїть, що_рухається) — ОДИН виклик, ОДНЕ тіло, ОДИН об'єкт.
    params   — значення параметра (≥2).

    Навіщо пара: справжня інваріантність і «параметр не дійшов до обчислення» дають
    однаковий вивід. Рухома величина доводить, що машинерія ЗДАТНА побачити рух — і
    лише тому «не зрушило» є результатом, а не сліпотою.

    ★ЧОМУ ОДНА ФУНКЦІЯ, А НЕ ДВІ (клас Альфи, S1059 ред.2, 2026-07-22): «мало МАТИ
    негконтроль — він мусить бити в ТОЙ САМИЙ детектор, що несе свідчення; інакше
    маємо два зелені, які нічого один про одного не знають». Саме це й сталося в
    S1059::T3, де основний тест міряв ненульовість члена ряду, а негконтроль —
    досяжність у r-просторі. Дві окремі функції дозволяли б порахувати контраст
    ІНШОЮ машинерією; повернення пари з одного тіла робить це неможливим за формою.
    """
    if len(params) < 2:
        raise ValueError("contrast requires ≥2 parameter values")
    got = [probe(p) for p in params]
    if any((not isinstance(g, tuple)) or len(g) != 2 for g in got):
        raise ValueError("probe(p) must return a PAIR (what_stands, what_moves) "
                         "from ONE computation — separate functions are not accepted by form")
    inv = [g[0] for g in got]
    mov = [g[1] for g in got]
    inv_holds = len(set(map(str, inv))) == 1
    mov_moves = len(set(map(str, mov))) > 1
    if not mov_moves:
        _void.append(msg)
        _say(f"  ☠ EMPTY  {msg}")
        _say(f"       the control quantity ALSO didn't move ({mov}) ⟹ the parameter never reached")
        _say(f"       the computation; the test reformulated the definition instead of measuring a response")
        return False
    (_pass if inv_holds else _fail).append(msg)
    _say(f"  {'✓' if inv_holds else '✗ FAIL'} {msg}")
    _say(f"       invariant {inv} ⊥ control {mov} — the same machinery MOVES the control,")
    _say(f"       so 'the invariant stands' is a measurement, not blindness")
    return inv_holds


def report(name="harness"):
    """Підсумок + exit-код. ☠ дає exit≠0 нарівні з ✗ — порожній assert є ПАДІННЯ.

    ★СЕРТИФІКАТ `FAILS: n` — І ЛИШЕ З ВЛАСНОГО ПРОГОНУ (вирок Ω 2026-08-08, сьома сліпота).
    Гейт `check_verify_green` читає рядок `FAILS: n` як засвідчення файла. Виміряно, що файл
    БЕЗ власного підсумку діставав зелений сертифікат від того, кого ІМПОРТУЄ: імпорт запускає
    батька, батько друкує свій рядок, дитина його успадковує. Тому рядок друкується ЛИШЕ коли
    `report()` викликано з модуля `__main__` — імпортований батько виконується під власним
    іменем, і його сертифікат не з'являється у виводі дитини.
    ⟨Межа названа: це лікує тих, хто засвідчується ЧЕРЕЗ харнес; віза, що друкує власний
    ЛІТЕРАЛ `FAILS:`, і далі може передати його імпортеру — там гейт МІРЯЄ факт завантаження
    й друкує позначку «сертифікат може бути успадкований».⟩
    """
    _say()
    _say(f"SCORE [{name}]: {len(_pass)} ✓ / {len(_fail)} ✗ / {len(_void)} ☠ EMPTY")
    if _void:
        _say("★EMPTY ASSERTS (not witnesses — remove or give them a negative world):")
        for m in _void:
            _say(f"   ☠ {m}")
    fails = len(_fail) + len(_void)
    if sys._getframe(1).f_globals.get("__name__") == "__main__":
        _say(f"FAILS: {fails}")
    return 1 if fails else 0


def reset():
    """Для самотесту: почати лічбу з нуля."""
    _pass.clear(); _fail.clear(); _void.clear()


# ───────────────────────────── САМОПЕРЕВІРКА ─────────────────────────────
def _selftest():
    """Прогін на ЧОТИРЬОХ реальних детекторах S1059. Очікування карбовані тут:
    T3 і T4-як-було МУСЯТЬ бути позначені порожніми, T2 і T4-змістовний — пройти."""
    import sympy as sp

    _say("=" * 78)
    _say("SELF-TEST `_teeth.py` — on real detectors from probe S1059")
    _say("=" * 78)

    g1, g2 = sp.symbols('g1 g2', nonzero=True)
    G0 = sp.Matrix([[g1, 0], [0, g2]])
    f, fb, q = sp.symbols('f fbar q', nonzero=True)

    def no_gaps(H):
        term = G0
        for _ in range(7):
            if sp.simplify(term.norm()) == 0:
                return False
            term = sp.simplify(term * H * G0)
        return True

    _say("\n[1] S1059::T3 «no omissions in the series» — GREEN as it stood in the probe")
    reset()
    ok(no_gaps, sp.Matrix([[0, f], [fb, 0]]),
       "T3: no term of the series n=0..6 is empty ⟹ the unit is not p≥2",
       must_fail_on=("a reach-2 operator — there the unit REALLY IS 2 edges",
                     sp.Matrix([[q, 0], [0, q]])))
    t3_void = len(_void) == 1

    _say("\n[2] S1059::T2 (algebraic route) — has teeth")
    reset()

    def no_local_root(pair):
        fp, fbp = pair
        num, _ = sp.fraction(sp.cancel(sp.together(-fp * fbp)))
        _, facs = sp.factor_list(sp.expand(num))
        return any(m % 2 == 1 and not b.is_Symbol for b, m in facs)

    X1, X2 = sp.symbols('X1 X2', nonzero=True)
    f2, fb2 = 1 + X1 + X2, 1 + 1 / X1 + 1 / X2
    ok(no_local_root, (f2, fb2),
       "T2: no local root K²=H exists (d=2)",
       must_fail_on=[("H² — there a local root DOES exist (H itself)", (f2 * fb2, f2 * fb2)),
                     ("f=1 — no node, a root is possible", (sp.Integer(1), sp.Integer(1)))])
    t2_pass = len(_pass) == 1 and not _void

    def hop_matrix(N, reach=1):
        A = sp.zeros(N, N)
        for i in range(N):
            A[i, (i + reach) % N] += 1
            A[(i + reach) % N, i] += 1
        return A

    def min_power(A, u, v, nmax=14):
        P = sp.eye(A.rows)
        for n in range(1, nmax + 1):
            P = P * A
            if sp.simplify(P[u, v]) != 0:
                return n
        return None

    _say("\n[3] S1059::T4 as it stood in the probe — P never reached the computation")
    reset()
    A_fixed = hop_matrix(12, 1)

    def probe_as_was(P):
        """A literal reconstruction: the matrix is built OUTSIDE the parameter, so both
        quantities read an object that P never touched."""
        return min_power(A_fixed, 0, 1), min_power(A_fixed, 0, 1)

    ok_contrast(probe_as_was, [2, 3, 4, 6], "T4-as-it-stood: per-bond count = 1 ∀P")
    t4_void = len(_void) == 1

    _say("\n[4] S1059::T4 in a meaningful form — P enters the CONSTRUCTION of the description")
    reset()

    def supercell(N, P):
        perm = [(i % P) * (N // P) + (i // P) for i in range(N)]
        base = hop_matrix(N, 1)
        A = sp.zeros(N, N)
        for i in range(N):
            for j in range(N):
                A[perm[i], perm[j]] = base[i, j]
        return A, perm

    def probe_meaningful(P):
        """ONE construction, ONE object: the same supercell matrix yields both
        quantities — otherwise 'invariant' and 'control' could read different worlds."""
        A, perm = supercell(12, P)
        return min_power(A, perm[0], perm[1]), min_power(A, perm[0], perm[P])

    ok_contrast(probe_meaningful, [2, 3, 4, 6],
                "T4-meaningful: the per-BOND count is invariant under relabelling P")
    t4_pass = len(_pass) == 1 and not _void

    _say()
    _say("=" * 78)
    _say("SELF-TEST VERDICT (expectations carved into the code, not read off the output):")
    checks = [("T3 flagged empty", t3_void),
              ("T2 passed with teeth", t2_pass),
              ("T4-as-it-stood flagged empty", t4_void),
              ("T4-meaningful passed with contrast", t4_pass)]
    for name, good in checks:
        _say(f"  {'✓' if good else '✗ FAIL'} {name}")
    bad = [n for n, g in checks if not g]
    if bad:
        _say(f"★HARNESS IS BROKEN: {bad} — it no longer catches what it exists to catch.")
    else:
        _say("★the harness catches exactly the two that passed green in the probe, and lets")
        _say("  through the two that have teeth. With no knowledge of physics whatsoever.")
    _say("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(_selftest())
