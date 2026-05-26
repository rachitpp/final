_COPY_ICON = (
    "<svg viewBox='0 0 24 24' width='14' height='14' fill='none' "
    "stroke='currentColor' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round' aria-hidden='true'>"
    "<rect x='9' y='9' width='13' height='13' rx='2'></rect>"
    "<path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'></path>"
    "</svg>"
)

ASSISTANT_ACTIONS_HTML = (
    "\n\n<div class='msg-actions'>"
    "<button class='copy-btn' type='button' aria-label='Copy answer'>"
    f"{_COPY_ICON}<span class='copy-label'>Copy</span>"
    "</button>"
    "</div>"
)

THINKING_HTML = (
    "<div class='thinking'>"
    "<span class='thinking-glyph' aria-hidden='true'>◐</span>"
    "<span class='thinking-stages'>"
    "<span class='stage stage-1'>Embedding query</span>"
    "<span class='stage stage-2'>Retrieving passages</span>"
    "<span class='stage stage-3'>Reranking results</span>"
    "<span class='stage stage-4'>Writing answer</span>"
    "</span>"
    "</div>"
)

ERROR_HTML = (
    "<div class='error-note'>"
    "<span class='error-mark'>!</span>"
    "<span>Something went wrong fetching that answer. "
    "Please try again in a moment.</span>"
    "</div>"
)
