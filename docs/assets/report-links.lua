-- Publication formatting: compact reference links, vector PDF figures and
-- a restrained, centered opening results heading in LaTeX output.
-- Run after citeproc. Body links and in-text citation links are unchanged.
function Image(img)
  if FORMAT:match("latex") then
    local publication_figures = {
      ["assets/kurversc-staged-architecture.svg"] = "assets/kurversc-staged-architecture.pdf",
      ["assets/kurversc-keyed-reduction.svg"] = "assets/kurversc-keyed-reduction.pdf",
      ["assets/kurversc-reduction-example.svg"] = "assets/kurversc-reduction-example.pdf",
      ["assets/kurversc-relarena-elo-20261105.svg"] = "assets/kurversc-relarena-elo-20261105.pdf",
    }
    if publication_figures[img.src] then
      img.src = publication_figures[img.src]
    end
  end
  return img
end

function Header(header)
  if FORMAT:match("latex") and header.identifier == "results-at-glance" then
    return {
      pandoc.RawBlock("latex", "\\begin{center}\n\\fontsize{11}{13}\\selectfont\\bfseries"),
      pandoc.Plain(header.content),
      pandoc.RawBlock("latex", "\\end{center}"),
    }
  end
end

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
