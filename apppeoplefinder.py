import os
import io
import json
import time
import re
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title='People Business Viability', layout='wide')
st.title('🔎 People Business Viability')
st.caption('Upload a CSV or paste email addresses, then clean/validate/dedup and optionally enrich results.')

# ---- Helpers ----
COUNTRY_DIAL_MAP = {
    '1': 'USA/Canada', '7': 'Russia/Kazakhstan', '20': 'Egypt', '27': 'South Africa',
    '30': 'Greece', '31': 'Netherlands', '32': 'Belgium', '33': 'France', '34': 'Spain',
    '36': 'Hungary', '39': 'Italy', '40': 'Romania', '41': 'Switzerland', '43': 'Austria',
    '44': 'United Kingdom', '45': 'Denmark', '46': 'Sweden', '47': 'Norway', '48': 'Poland',
    '49': 'Germany', '51': 'Peru', '52': 'Mexico', '53': 'Cuba', '54': 'Argentina', '55': 'Brazil',
    '56': 'Chile', '57': 'Colombia', '58': 'Venezuela', '60': 'Malaysia', '61': 'Australia',
    '62': 'Indonesia', '63': 'Philippines', '64': 'New Zealand', '65': 'Singapore', '66': 'Thailand',
    '81': 'Japan', '82': 'South Korea', '84': 'Vietnam', '86': 'China', '90': 'Turkey',
    '91': 'India', '92': 'Pakistan', '93': 'Afghanistan', '94': 'Sri Lanka', '95': 'Myanmar',
    '98': 'Iran', '211': 'South Sudan', '212': 'Morocco', '213': 'Algeria', '216': 'Tunisia',
    '218': 'Libya', '220': 'Gambia', '221': 'Senegal', '222': 'Mauritania', '223': 'Mali',
    '224': 'Guinea', '225': "Côte d'Ivoire", '226': 'Burkina Faso', '227': 'Niger', '228': 'Togo',
    '229': 'Benin', '230': 'Mauritius', '231': 'Liberia', '232': 'Sierra Leone', '233': 'Ghana',
    '234': 'Nigeria', '235': 'Chad', '236': 'Central African Republic', '237': 'Cameroon',
    '238': 'Cape Verde', '239': 'São Tomé and Príncipe', '240': 'Equatorial Guinea',
    '241': 'Gabon', '242': 'Republic of the Congo', '243': 'DR Congo', '244': 'Angola',
    '245': 'Guinea-Bissau', '246': 'British Indian Ocean Territory', '248': 'Seychelles',
    '249': 'Sudan', '250': 'Rwanda', '251': 'Ethiopia', '252': 'Somalia', '253': 'Djibouti',
    '254': 'Kenya', '255': 'Tanzania', '256': 'Uganda', '257': 'Burundi', '258': 'Mozambique',
    '260': 'Zambia', '261': 'Madagascar', '262': 'Réunion/Mayotte', '263': 'Zimbabwe',
    '264': 'Namibia', '265': 'Malawi', '266': 'Lesotho', '267': 'Botswana', '268': 'Eswatini',
    '269': 'Comoros', '290': 'Saint Helena', '291': 'Eritrea', '297': 'Aruba', '298': 'Faroe Islands',
    '299': 'Greenland', '350': 'Gibraltar', '351': 'Portugal', '352': 'Luxembourg',
    '353': 'Ireland', '354': 'Iceland', '355': 'Albania', '356': 'Malta', '357': 'Cyprus',
    '358': 'Finland', '359': 'Bulgaria', '370': 'Lithuania', '371': 'Latvia', '372': 'Estonia',
    '373': 'Moldova', '374': 'Armenia', '375': 'Belarus', '376': 'Andorra', '377': 'Monaco',
    '378': 'San Marino', '380': 'Ukraine', '381': 'Serbia', '382': 'Montenegro', '383': 'Kosovo',
    '385': 'Croatia', '386': 'Slovenia', '387': 'Bosnia and Herzegovina', '389': 'North Macedonia',
    '420': 'Czechia', '421': 'Slovakia', '423': 'Liechtenstein', '500': 'Falkland Islands',
    '501': 'Belize', '502': 'Guatemala', '503': 'El Salvador', '504': 'Honduras', '505': 'Nicaragua',
    '506': 'Costa Rica', '507': 'Panama', '509': 'Haiti', '590': 'Guadeloupe', '591': 'Bolivia',
    '592': 'Guyana', '593': 'Ecuador', '594': 'French Guiana', '595': 'Paraguay', '596': 'Martinique',
    '597': 'Suriname', '598': 'Uruguay', '599': 'Caribbean Netherlands', '670': 'Timor-Leste',
    '672': 'Norfolk Island', '673': 'Brunei', '674': 'Nauru', '675': 'Papua New Guinea',
    '676': 'Tonga', '677': 'Solomon Islands', '678': 'Vanuatu', '679': 'Fiji', '680': 'Palau',
    '681': 'Wallis and Futuna', '682': 'Cook Islands', '683': 'Niue', '685': 'Samoa',
    '686': 'Kiribati', '687': 'New Caledonia', '688': 'Tuvalu', '689': 'French Polynesia',
    '690': 'Tokelau', '691': 'Micronesia', '692': 'Marshall Islands', '850': 'North Korea',
    '852': 'Hong Kong', '853': 'Macau', '855': 'Cambodia', '856': 'Laos', '880': 'Bangladesh',
    '886': 'Taiwan', '960': 'Maldives', '961': 'Lebanon', '962': 'Jordan', '963': 'Syria',
    '964': 'Iraq', '965': 'Kuwait', '966': 'Saudi Arabia', '967': 'Yemen', '968': 'Oman',
    '970': 'Palestine', '971': 'UAE', '972': 'Israel', '973': 'Bahrain', '974': 'Qatar',
    '975': 'Bhutan', '976': 'Mongolia', '977': 'Nepal', '992': 'Tajikistan', '993': 'Turkmenistan',
    '994': 'Azerbaijan', '995': 'Georgia', '996': 'Kyrgyzstan', '998': 'Uzbekistan'
}

def cc_to_country(code):
    if pd.isna(code):
        return ''
    s = str(code).strip().replace('+','')
    s = re.sub(r'[^0-9]', '', s)
    if not s:
        return ''
    for k in sorted(COUNTRY_DIAL_MAP.keys(), key=lambda x: -len(x)):
        if s.startswith(k):
            return COUNTRY_DIAL_MAP[k]
    return f'Unknown (+{s})'

def valid_full_name(first, last):
    if pd.isna(first) or pd.isna(last):
        return False
    full = str(first).strip() + str(last).strip()
    return len(full) > 2

def valid_email(email):
    if pd.isna(email):
        return False
    email = str(email).strip()
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email))

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in list(df.columns):
        low = c.strip().lower()
        if low in {'first name','firstname','first'}: rename_map[c] = 'first_name'
        elif low in {'last name','lastname','last'}: rename_map[c] = 'last_name'
        elif low in {'email','emails'}: rename_map[c] = 'email'
        elif low in {'country','location'}: rename_map[c] = 'country'
        elif low in {'phone country code','dial_code'}: rename_map[c] = 'phone_country_code'
        elif low in {'session'}: rename_map[c] = 'session'
        elif low in {'blurb','profile_blurb'}: rename_map[c] = 'blurb'
        elif low in {'sources','urls'}: rename_map[c] = 'sources'
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in ['first_name','last_name','email','country','phone_country_code','session','blurb','sources']:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def parse_session_value(x):
    if pd.isna(x) or str(x).strip().lower() in {'nan','none',''}:
        return pd.NA
    s = str(x).strip()
    ts = pd.to_datetime(s, errors='coerce', infer_datetime_format=True, dayfirst=False)
    if pd.isna(ts):
        try:
            from datetime import datetime
            ts = datetime.strptime(s.replace(',',''), '%a %d %b %Y %I:%M %p')
        except Exception:
            return s
    return ts.strftime('%Y-%m-%d %H:%M')

def dedup_by_email_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    if 'session' in df.columns:
        parsed = df['session'].apply(parse_session_value)
        df = df.assign(_session_sort=parsed)
        df = df.sort_values(by=['email','_session_sort'], ascending=[True, True])
        return df.drop_duplicates(subset=['email'], keep='last').drop(columns=['_session_sort'])
    return df.drop_duplicates(subset=['email'], keep='last')

SYSTEM_PROMPT = ('You are an assistant that evaluates if a person appears relevant to a given business question '
                 'based on their profile blurb and URLs. Consider the provided company/brand and the user question. '
                 'Return only a compact JSON with keys: full_name, email, relevance (yes/no/maybe), '
                 'category, rationale (1-2 sentences), and sources (up to 5 URLs).')
DEFAULT_MODEL = 'gpt-4o-mini'
TEMP = 0.2
MAX_TOKENS = 400

def call_openai_enrichment(full_name, email, country, blurb, sources, company, question, model=DEFAULT_MODEL):
    if OpenAI is None:
        raise RuntimeError('openai package not available. Install `openai`.')
    api_key = os.getenv('OPENAI_API_KEY') or st.session_state.get('OPENAI_API_KEY_UI')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set. Set env var or enter in the sidebar.')
    client = OpenAI(api_key=api_key)
    user_input = {
        'company': company or '',
        'question': question or '',
        'profile_blurb': blurb or '',
        'sources': [s for s in re.split(r'[\n\|,\s]+', sources or '') if s.startswith('http')],
        'person': {'full_name': full_name or '', 'email': email or '', 'country': country or ''}
    }
    msg = json.dumps(user_input)
    resp = client.responses.create(
        model=model,
        input=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':msg}],
        temperature=TEMP,
        max_output_tokens=MAX_TOKENS
    )
    try:
        text = resp.output_text
    except Exception:
        text = str(resp)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {'full_name': full_name or '', 'email': email or '', 'relevance': 'maybe', 'category': 'unknown',
            'rationale': (blurb or '')[:180],
            'sources': [s for s in re.split(r'[\n\|,\s]+', sources or '') if s.startswith('http')][:5]}

def dataframe_to_jsonl(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    for _, row in df.iterrows():
        buf.write(json.dumps({k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}, ensure_ascii=False) + '\n')
    return buf.getvalue()

def parse_emails_block(block: str):
    """Return a de-duplicated list of email addresses from a raw block (comma/space/newline separated)."""
    if not block:
        return []
    found = re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', block)
    # normalize and dedup preserving order
    seen = set()
    result = []
    for e in found:
        e2 = e.strip()
        if e2 and e2.lower() not in seen:
            seen.add(e2.lower())
            result.append(e2)
    return result

# ---- Sidebar (global settings) ----
with st.sidebar:
    st.header('Settings')
    st.write('Optional: provide an OpenAI API key to enable enrichment.')
    api_key_input = st.text_input('OpenAI API Key', type='password', value=os.getenv('OPENAI_API_KEY') or '')
    if api_key_input:
        st.session_state['OPENAI_API_KEY_UI'] = api_key_input
    model_choice = st.selectbox('Model', ['gpt-4o-mini','gpt-4.1-mini','gpt-4o'], index=0)
    company = st.text_input('Your company/brand (used in reasoning)', value='')
    question = st.text_area('What do you want to know about these people?', help='e.g., Will they attend my workshop? Is product X a good fit?')

tabs = st.tabs(['📁 Upload CSV','📧 Paste Email Address(es)','ℹ️ Help'])

# ---- Tab 1: CSV Upload ----
with tabs[0]:
    st.subheader('Upload a CSV of people')
    st.write('Expected columns: first name, last name, email, (optional) country/phone code, blurb, sources.')
    file = st.file_uploader('Choose CSV file', type=['csv'])
    if file is not None:
        df = pd.read_csv(file)
        df = normalize_columns(df)
        if 'phone_country_code' in df.columns:
            df['country'] = df['country'].fillna('')
            df['country'] = df.apply(lambda r: r['country'] or cc_to_country(r['phone_country_code']), axis=1)
        mask = df.apply(lambda r: valid_full_name(r['first_name'], r['last_name']), axis=1) & df['email'].apply(valid_email)
        cleaned = df[mask].copy()
        cleaned = dedup_by_email_keep_last(cleaned)
        st.session_state['cleaned_df'] = cleaned
        st.success(f'Cleaned rows: {len(cleaned)} of {len(df)}')
        st.dataframe(cleaned, use_container_width=True)
        st.subheader('Downloads')
        st.download_button('Download Cleaned CSV', cleaned.to_csv(index=False).encode('utf-8'), file_name='cleaned_profiles.csv', mime='text/csv')
        # Enrichment button
        if st.button('🚀 Run AI enrichment on cleaned rows'):
            if len(cleaned) == 0:
                st.warning('No rows to enrich.')
            else:
                st.info('Running enrichment...')
                results = []
                progress = st.progress(0)
                rows = cleaned.to_dict(orient='records')
                for i, r in enumerate(rows, start=1):
                    full_name = f"{(r.get('first_name') or '').strip()} {(r.get('last_name') or '').strip()}".strip()
                    out = call_openai_enrichment(
                        full_name=full_name,
                        email=r.get('email') or '',
                        country=r.get('country') or '',
                        blurb=r.get('blurb') or '',
                        sources=r.get('sources') or '',
                        company=company,
                        question=question,
                        model=model_choice
                    )
                    results.append(out)
                    progress.progress(i/len(rows))
                    time.sleep(0.05)
                enriched_df = pd.DataFrame(results)
                enriched_df.insert(0, 'pipeline_source', 'csv_upload')
                st.subheader('Enriched Results')
                st.dataframe(enriched_df, use_container_width=True)
                st.download_button('Download Enriched CSV', enriched_df.to_csv(index=False).encode('utf-8'), file_name='enriched_profiles.csv', mime='text/csv')
                st.download_button('Download Enriched JSONL', dataframe_to_jsonl(enriched_df).encode('utf-8'), file_name='enriched_profiles.jsonl', mime='application/json')

# ---- Tab 2: Paste email addresses ----
with tabs[1]:
    st.subheader('Paste one or many email addresses')
    st.write('Enter emails separated by commas, spaces, or new lines. We will run the same enrichment for each email.')
    emails_block = st.text_area('Email address(es)', height=160, placeholder='alice@example.com, bob@domain.org\ncarol@site.io')
    parsed_emails = parse_emails_block(emails_block)
    if parsed_emails:
        st.success(f'Found {len(parsed_emails)} unique email(s).')
        st.write(parsed_emails)
    # Enrichment button for pasted emails
    if st.button('🚀 Run AI enrichment on pasted email(s)'):
        if not parsed_emails:
            st.warning('Please paste at least one email address.')
        else:
            st.info('Running enrichment...')
            results = []
            progress = st.progress(0)
            for i, em in enumerate(parsed_emails, start=1):
                out = call_openai_enrichment(
                    full_name='',
                    email=em,
                    country='',
                    blurb='',
                    sources='',
                    company=company,
                    question=question,
                    model=model_choice
                )
                results.append(out)
                progress.progress(i/len(parsed_emails))
                time.sleep(0.05)
            enriched_df2 = pd.DataFrame(results)
            enriched_df2.insert(0, 'pipeline_source', 'pasted_emails')
            st.subheader('Enriched Results (pasted)')
            st.dataframe(enriched_df2, use_container_width=True)
            st.download_button('Download Enriched CSV (pasted)', enriched_df2.to_csv(index=False).encode('utf-8'), file_name='enriched_from_pasted.csv', mime='text/csv')
            st.download_button('Download Enriched JSONL (pasted)', dataframe_to_jsonl(enriched_df2).encode('utf-8'), file_name='enriched_from_pasted.jsonl', mime='application/json')

# ---- Tab 3: Help ----
with tabs[2]:
    st.markdown(
        """
**How to use**
1. **Upload CSV**: Provide columns like *first name*, *last name*, *email*, optional *country/phone code*, *blurb*, *sources*.
   - The app cleans/validates/dedups and lets you download the cleaned CSV.
   - Click **Run AI enrichment** to score relevance and generate rationales & URLs.
2. **Paste Email Address(es)**: Paste one or many email addresses; click **Run AI enrichment** to evaluate each using the same prompt.

**Company & Question**
- In the sidebar, set your **Company/Brand** and **What you want to know** (e.g., workshop attendance, product fit).
- These guide the model’s reasoning and are included in the prompt.

**Outputs**
- Enriched outputs include a leading `pipeline_source` column indicating where the row came from.
        """
    )