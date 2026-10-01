"""Matplotlib text as LaTeX for the tikzfigure backend.

Matplotlib text is plain text with optional ``$...$`` mathtext. Plain text
is escaped for LaTeX (``_``, ``%``, ``&`` and so on are literal in
Matplotlib); mathtext is passed on, being nearly LaTeX already, without
Matplotlib's own commands such as ``\\mathdefault``. Unicode symbols
(``ω``, ``−``, ``°``) become math, which pdflatex can typeset.
"""

import re

# Unicode characters pdflatex does not typeset by default, as math.
_UNICODE_MATH = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\varepsilon",
    "ϵ": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ϑ": r"\vartheta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "µ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\varphi",
    "ϕ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "−": "-",
    "±": r"\pm",
    "∓": r"\mp",
    "×": r"\times",
    "·": r"\cdot",
    "÷": r"\div",
    "°": r"^\circ",
    "≈": r"\approx",
    "∼": r"\sim",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "∝": r"\propto",
    "∞": r"\infty",
    "∂": r"\partial",
    "∇": r"\nabla",
    "∫": r"\int",
    "∑": r"\sum",
    "√": r"\surd",
    "‖": r"\|",
    "∥": r"\parallel",
    "⊥": r"\perp",
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "ℏ": r"\hbar",
    "ℓ": r"\ell",
    "′": r"\prime",
    "⊙": r"\odot",
    "⊗": r"\otimes",
    "⁰": "^{0}",
    "¹": "^{1}",
    "²": "^{2}",
    "³": "^{3}",
    "⁴": "^{4}",
    "⁵": "^{5}",
    "⁶": "^{6}",
    "⁷": "^{7}",
    "⁸": "^{8}",
    "⁹": "^{9}",
    "⁻": "^{-}",
    "₀": "_{0}",
    "₁": "_{1}",
    "₂": "_{2}",
    "₃": "_{3}",
}

_TEXT_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "$": r"\$",
}

# mathtext commands that LaTeX lacks, and what they mean there
_MATH_COMMANDS = [
    (re.compile(r"\\mathdefault\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\mathdefault\b"), ""),
    (re.compile(r"\\degree\b"), r"{^\\circ}"),
    (re.compile(r"\\AA\b"), r"\\text{\\AA}"),
]


def _math(source: str) -> str:
    for pattern, replacement in _MATH_COMMANDS:
        source = pattern.sub(replacement, source)
    out = []
    for char in source:
        symbol = _UNICODE_MATH.get(char)
        if symbol is None:
            out.append(char)
        elif symbol.startswith("\\") and symbol[-1:].isalpha():
            out.append(symbol + " ")
        else:
            out.append(symbol)
    return "".join(out)


def _text(source: str) -> str:
    out = []
    for char in source:
        if char in _TEXT_ESCAPES:
            out.append(_TEXT_ESCAPES[char])
        elif char in _UNICODE_MATH:
            out.append(f"${_UNICODE_MATH[char]}$")
        elif char == "\n":
            out.append(r"\\")
        else:
            out.append(char)
    return "".join(out)


def latex(text) -> str:
    r"""Matplotlib text as LaTeX: plain parts escaped, ``$...$`` parts as math.

    An odd number of unescaped ``$`` makes Matplotlib show the text as it
    is, so then every part is plain text.

    Examples
    --------
    >>> latex(r"growth rate $\gamma/\omega_{ci}$ (50% fit)")
    'growth rate $\\gamma/\\omega_{ci}$ (50\\% fit)'
    >>> latex(r"$\mathdefault{0.5}$")
    '$0.5$'
    >>> latex("ω = 2.5")
    '$\\omega$ = 2.5'
    """
    if text is None:
        return ""
    text = str(text)
    parts = re.split(r"(?<!\\)\$", text)
    if len(parts) % 2 == 0:  # unbalanced: no math
        return _text(text.replace(r"\$", "$"))
    out = []
    for index, part in enumerate(parts):
        if index % 2:
            out.append(f"${_math(part)}$")
        else:
            out.append(_text(part.replace(r"\$", "$")))
    return "".join(out)


def is_multiline(text) -> bool:
    """Whether ``text`` has more than one line, needing ``align`` in TikZ."""
    return "\n" in str(text or "")
