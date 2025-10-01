-- Pandoc Lua filter to convert labeled display math to numbered equations
-- This enables automatic equation numbering for equations with \label{eq:...}

function Para(el)
  -- Check if paragraph contains only display math
  if #el.content == 1 and el.content[1].t == 'Math' and el.content[1].mathtype == 'DisplayMath' then
    local math_content = el.content[1].text
    -- Check if equation has a label
    if math_content:match('\\label{eq:') then
      -- Convert to equation environment for numbering
      local raw_latex = '\\begin{equation}\n' .. math_content .. '\n\\end{equation}'
      return pandoc.RawBlock('latex', raw_latex)
    end
  end
  return el
end
