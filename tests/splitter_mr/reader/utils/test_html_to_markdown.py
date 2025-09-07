import re

from splitter_mr.reader.utils.html_to_markdown import HtmlToMarkdown

# ---- Helpers, mocks and fixtures ---- #

HTML_SAMPLE: str = """
<h1 id="sample-markdown">Sample Markdown</h1>
<p>This is some basic, sample markdown.</p>
<h2 id="second-heading">Second Heading</h2>
<ul>
  <li>Unordered lists, and:
    <ol>
      <li>One</li>
      <li>Two</li>
      <li>Three</li>
    </ol>
  </li>
  <li>More</li>
</ul>
<blockquote>
  <p>Blockquote</p>
</blockquote>
<p>And <strong>bold</strong>, <em>italics</em>, and even <em>italics and later <strong>bold</strong></em>. Even <del>strikethrough</del>. <a href="https://markdowntohtml.com">A link</a> to somewhere.</p>
<p>And code highlighting:</p>
<pre><code class="lang-js">var foo = 'bar';

function baz(s) {
   return foo + ':' + s;
}
</code></pre>
<p>Or inline code like <code>var foo = &#39;bar&#39;;</code>.</p>
<p>Or an image of bears</p>
<p><img src="http://placebear.com/200/200" alt="bears"></p>
<p>The end ...</p>
"""

# ---- Test cases ---- #


def test_headings_paragraphs_and_blockquote():
    md = HtmlToMarkdown().convert(HTML_SAMPLE)

    # h1 / h2
    assert "# Sample Markdown" in md
    assert "## Second Heading" in md

    # paragraph text
    assert "This is some basic, sample markdown." in md

    # blockquote
    assert "> Blockquote" in md


def test_lists_ul_ol_and_nesting_basic_presence():
    md = HtmlToMarkdown().convert(HTML_SAMPLE)

    # Bullet for first li
    assert re.search(r"^- Unordered lists, and:", md, flags=re.M) is not None
    # Ordered items appear
    assert "1. One" in md
    assert "2. Two" in md
    assert "3. Three" in md
    # Second bullet
    assert re.search(r"^- More$", md, flags=re.M) is not None


def test_inline_styles_link_and_no_spurious_newlines_around_inline():
    md = HtmlToMarkdown().convert(HTML_SAMPLE)

    # Bold, italics, strike
    assert "**bold**" in md
    assert "*italics*" in md
    # nested strong inside em
    assert "*italics and later **bold***" in md

    # link
    assert "[A link](https://markdowntohtml.com)" in md

    # Ensure commas and inline pieces aren't split by extra blank lines
    assert (
        "And **bold**, *italics*, and even *italics and later **bold***. Even ~~strikethrough~~."
        in md
    )


def test_code_block_language_and_inline_code():
    md = HtmlToMarkdown().convert(HTML_SAMPLE)

    # Code fence with language from class=lang-js
    assert "```js" in md
    assert "var foo = 'bar';" in md
    assert "function baz(s)" in md
    assert md.count("```") >= 2  # open + close

    # Inline code is backticked and entity-decoded
    assert "Or inline code like `var foo = 'bar';`." in md


def test_image_markdown():
    md = HtmlToMarkdown().convert(HTML_SAMPLE)
    assert "![bears](http://placebear.com/200/200)" in md


def test_table_with_thead_to_markdown():
    html = """
    <table>
      <thead><tr><th>H1</th><th>H2</th></tr></thead>
      <tbody>
        <tr><td>A1</td><td>A2</td></tr>
        <tr><td>B1</td><td>B2</td></tr>
      </tbody>
    </table>
    """
    md = HtmlToMarkdown().convert(html).strip().splitlines()

    # header line + separator + two data lines
    assert md[0] == "| H1 | H2 |"
    assert md[1] == "| --- | --- |"
    assert "| A1 | A2 |" in md
    assert "| B1 | B2 |" in md


def test_table_without_thead_uses_first_row_as_header():
    html = """
    <table>
      <tr><th>Name</th><th>Age</th></tr>
      <tr><td>Alice</td><td>30</td></tr>
      <tr><td>Bob</td><td>40</td></tr>
    </table>
    """
    md = HtmlToMarkdown().convert(html).strip().splitlines()
    assert md[0] == "| Name | Age |"
    assert md[1] == "| --- | --- |"
    assert "| Alice | 30 |" in md
    assert "| Bob | 40 |" in md


def test_pre_without_code_class_no_language_marker():
    html = """
    <pre><code>echo "hello &lt;world&gt;"</code></pre>
    """
    md = HtmlToMarkdown().convert(html)
    # no language label after ```
    assert "```" in md and "```" + "\n" in md
    assert 'echo "hello <world>"' in md


def test_entity_unescaping_and_wrapper_flattening():
    html = """
    <section>
      <div>
        <p>AT&amp;T &lt;Corp&gt; &quot;quoted&quot;</p>
      </div>
    </section>
    """
    md = HtmlToMarkdown().convert(html)
    assert 'AT&T <Corp> "quoted"' in md


def test_blank_line_collapse_and_strip():
    html = """
    <div>
      <p>One</p>
      <p></p>
      <p>Two</p>
      <p>   </p>
      <p>Three</p>
    </div>
    """
    md = HtmlToMarkdown().convert(html)
    # No triple+ blank newlines
    assert re.search(r"\n{3,}", md) is None
    # Still contains the three lines of text in order
    assert "One" in md and "Two" in md and "Three" in md


def test_list_indentation_nested_ul():
    html = """
    <ul>
      <li>Parent
        <ul>
          <li>Child A</li>
          <li>Child B</li>
        </ul>
      </li>
    </ul>
    """
    md = HtmlToMarkdown().convert(html)
    # top-level bullet
    assert re.search(r"^- Parent", md, flags=re.M)
    # child bullets should appear somewhere below
    assert "Child A" in md
    assert "Child B" in md


def test_empty_and_misc_edges():
    # Empty string → empty
    assert HtmlToMarkdown().convert("") == ""

    # Unknown tags should fall back to contents
    html = "<custom><p>Hello</p><span>World</span></custom>"
    md = HtmlToMarkdown().convert(html)
    assert "Hello" in md and "World" in md
