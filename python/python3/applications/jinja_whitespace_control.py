#!/usr/bin/env python3

# Whitespace control with jinja templating.
# https://jinja.palletsprojects.com/en/stable/templates/#whitespace-control

import jinja2

env = jinja2.Environment()


temp_str="""
# -----------------------------------------------------------------------------
# Default ---------------------------------------------------------------------
{% for i in range(3) %}
    line {{i}}
{% endfor %}
{% if VAR_TRUE %}
    true!
{% endif %}
{% if VAR_FALSE %}
    false!
{% endif %}
# -----------------------------------------------------------------------------
# Minus at end of start of block: trims heading newline, whitespace  ----------
{% for i in range(3) -%}
    line {{i}}
{% endfor %}
{% if VAR_TRUE -%}
    true!
{% endif %}
{% if VAR_FALSE -%}
    false!
{% endif %}
# -----------------------------------------------------------------------------
# Minus at start of end of block: trims trailing newline, whitespace  ---------
{% for i in range(3) %}
    line {{i}}
{%- endfor %}
{% if VAR_TRUE %}
    true! (note the missing newline below compared to outputs above)
{%- endif %}
{% if VAR_FALSE %}
    false!
{%- endif %}
"""

temp = env.from_string(temp_str)
print(temp.render(VAR_TRUE=True, VAR_FALSE=False))

