-- Compact bibliography link labels; preserve all original link destinations.
-- Run after citeproc. Body links and in-text citation links are unchanged.
local function compact_link(link)
  local target = link.target
  local arxiv = target:match("arxiv%.org/abs/([^?#]+)")
    or target:match("arxiv%.org/html/([^?#]+)")
  local label
  if arxiv then
    label = "arXiv:" .. arxiv
  elseif target:match("^https?://doi%.org/") then
    label = target:gsub("^https?://doi%.org/", "")
  elseif target:match("^https?://github%.com/") then
    label = "GitHub source"
  elseif target:match("^https?://web%.archive%.org/") then
    label = "Archived report"
  elseif target:match("^https?://docs%.nvidia%.com/") then
    label = "NVIDIA Kumo documentation"
  elseif target:match("^https?://kurveai%.github%.io/") then
    label = "Kurve documentation"
  elseif target:match("^https?://kurve%.ai/") then
    label = "Kurve AI"
  elseif target:match("^https?://proceedings%.")
    or target:match("^https?://jmlr%.org/") then
    label = "Publisher"
  elseif target:match("%.pdf$") then
    label = "Paper PDF"
  else
    label = target:match("^https?://([^/]+)")
  end
  if label then
    link.content = pandoc.Inlines({pandoc.Str(label)})
  end
  return link
end

function Div(div)
  if div.identifier == "refs" then
    return div:walk({Link = compact_link})
  end
end
