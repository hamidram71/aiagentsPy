# راه‌اندازی قدم‌به‌قدم

هفت قدم، از صفر تا یک گزارش واقعی. هر قدم **خروجی مورد انتظار** دارد و
**اگر خطا داد چه کنید**. قدم‌های ۱ تا ۴ هیچ کلید API و هیچ هزینه‌ای ندارند.

---

## قدم ۱ — پیش‌نیازها

```bash
python --version
```

باید **۳.۱۰ یا بالاتر** باشد (پروژه روی ۳.۱۱ تست شده). اگر `python` پیدا
نشد، `python3` را امتحان کنید و در بقیهٔ دستورها هم همان را به کار ببرید.

اگر نسخه‌تان قدیمی‌تر است: کد از سینتکس `str | None` استفاده می‌کند که در
۳.۹ خطا می‌دهد. پایتون جدیدتر نصب کنید.

---

## قدم ۲ — محیط مجازی و نصب

فایل زیپ را باز کنید و داخل پوشه بروید:

```bash
cd csv-insight-agent
```

**لینوکس / مک:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**ویندوز (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

اگر PowerShell اجازه نداد (`running scripts is disabled`):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

بعد نصب:

```bash
pip install -r requirements.txt
```

**خروجی مورد انتظار:** چند خط دانلود و در پایان `Successfully installed ...`.
حدود ۲ تا ۴ دقیقه طول می‌کشد.

از این به بعد، هر بار که ترمینال جدید باز می‌کنید باید محیط مجازی را
دوباره فعال کنید. اگر `ModuleNotFoundError: No module named 'langgraph'`
گرفتید، معمولاً یعنی همین را فراموش کرده‌اید — نه این‌که نصب خراب است.

---

## قدم ۳ — تست‌ها (بدون کلید API)

```bash
pytest
```

**خروجی مورد انتظار:**

```
........................................................................ [ 31%]
...
242 passed, 1 skipped in 100.62s      ← ویندوز
243 passed in 45.12s                  ← لینوکس / مک
```

عدد ثانیه‌ها فرق می‌کند. آن `1 skipped` روی ویندوز طبیعی است: تستی که
Ctrl+C را با فرستادن سیگنال به یک پروسهٔ دیگر امتحان می‌کند، روی ویندوز قابل
اجرا نیست. اگر یکی از providerها (`langchain-anthropic` یا `langchain-openai`)
را نصب نکرده باشید، چند skip دیگر هم اضافه می‌شود؛ `pytest -rs` می‌گوید کدام و چرا.

این مهم‌ترین قدم است. اگر همهٔ تست‌ها سبز شد، محیط شما درست است و کل
پلامبینگ ایجنت — reducerها، مسیریابی، سیم‌کشی ابزارها، سندباکس، توقف و
ادامه — سالم کار می‌کند.

**اگر خطا داد:**

| خطا | معنی |
|---|---|
| `ModuleNotFoundError: No module named 'dotenv'` (یا هر پکیج دیگر) | پکیج در **همین** venv نصب نیست → `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError` | محیط مجازی فعال نیست، یا `pip install` تمام نشده |
| `pytest: command not found` | `python -m pytest` را امتحان کنید |
| `FileNotFoundError: data/sales.csv` | از داخل پوشهٔ `csv-insight-agent` اجرا نکرده‌اید |
| `s` به‌جای `.` دیدید | skip است، نه خطا. روی ویندوز طبیعی است |
| `F` در `test_llm` | پکیج یکی از providerها نصب نیست — از نسخهٔ جدید پروژه استفاده کنید، تست‌ها حالا skip می‌کنند |

برای دیدن اسم تست‌ها یکی‌یکی:

```bash
pytest -v
```

و اگر خواستید فقط یک بخش را ببینید:

```bash
pytest tests/test_sandbox.py -v      # فقط سندباکس
pytest tests/test_graph.py -v        # فقط گراف و مسیریابی
```

---

## قدم ۴ — اجرای خشک با مدل اسکریپت‌شده (بدون هزینه)

اینجا کل گراف اجرا می‌شود، ولی جواب‌های مدل از یک صف آماده می‌آیند. یعنی
pandas واقعی، نمودار واقعی، فایل واقعی — و صفر تومان هزینه.

```bash
python main.py data/sales.csv "Which region drives revenue?" --provider fake
```

روی سؤال تأیید، `approve` بزنید و Enter.

**خروجی مورد انتظار:**

```
thread: run-a1b2c3d4
provider: fake / fake
question: Which region drives revenue?

  [loader] loaded 425 rows x 12 columns
  [planner] plan:
        1. Total revenue per region
        2. Monthly revenue trend
        3. Write the report

==============================================================
PLAN APPROVAL NEEDED
  1. Total revenue per region
  ...
==============================================================
approve, or describe what to change > approve
  [supervisor] Supervisor -> analyst: Revenue per region (...)
  [analyst] tool -> group_aggregate({'group_by': ['region'], ...})
  [analyst] finding: North is the largest region by revenue.
  [supervisor] Supervisor -> charter: Bar chart of revenue per region (...)
  [charter] chart:   .../outputs/revenue_by_region.png
  [supervisor] Supervisor -> reporter: Write it up (...)
  [reporter] report -> .../outputs/report_20260915_084624.md
  [supervisor] Supervisor: report exists, finishing.
```

بعد نگاه کنید داخل پوشه چه ساخته شده:

```bash
ls outputs/
```

باید سه چیز ببینید: یک `report_*.md`، یک `revenue_by_region.png`، و
`checkpoints.sqlite`. فایل PNG را باز کنید — نمودار واقعی است، از دادهٔ
واقعی.

### توقف و ادامه را امتحان کنید

این قشنگ‌ترین بخش پروژه است. یک اجرا شروع کنید و **روی سؤال تأیید، ترمینال
را ببندید** (یا `Ctrl+C` بزنید):

```bash
python main.py data/sales.csv "test" --provider fake --thread-id demo
```

بعد در ترمینال جدید:

```bash
python main.py --resume --thread-id demo --provider fake
```

گراف از همان نقطه ادامه می‌دهد. state در `outputs/checkpoints.sqlite`
زنده مانده بود، نه در حافظهٔ آن پروسه.

### نقشهٔ گراف را ببینید

```bash
python draw_graph.py
```

متن Mermaid چاپ می‌شود. کپی کنید و در <https://mermaid.live> بریزید تا
نقشه را ببینید. ارزش دارد یک‌بار ببینیدش قبل از خواندن کد.

---

## قدم ۵ — تنظیم کلید API

فایل نمونه را کپی کنید:

```bash
cp .env.example .env         # ویندوز: copy .env.example .env
```

`.env` را با یک ویرایشگر متن باز کنید. **حداقل کاری که لازم است** یکی از
این دو حالت است:

**برای OpenAI:**

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-6-luna
OPENAI_API_KEY=sk-...کلید واقعی شما...
```

**برای Claude:**

```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-5
ANTHROPIC_API_KEY=sk-ant-...کلید واقعی شما...
```

کلید را از پنل خود provider می‌گیرید (`platform.openai.com` یا
`console.anthropic.com`). `.env` در `.gitignore` هست، پس اگر پروژه را روی
git گذاشتید کلیدتان بیرون نمی‌رود — **ولی یک‌بار خودتان چک کنید**، این
دقیقاً همان اشتباهی است که همه یک‌بار مرتکب می‌شوند.

`MAX_STEPS` را دست نزنید؛ پیش‌فرض (۳۰) را نگه دارید. نسخه‌های قبلی این راهنما
می‌گفتند برای اجرای اول `MAX_STEPS=8` بگذارید تا ارزان بماند. با مدل واقعی
این **تضمین** می‌کند که جواب نیمه‌کاره بماند: هر قدم نقشه سه چهار نوبت مدل
می‌خورد و بودجه قبل از جواب دادن به سؤال تمام می‌شود. هزینهٔ یک اجرا با
`MAX_TOOL_ROUNDS` (پیش‌فرض ۴ نوبت برای هر قدم) محدود است، نه با گرسنه نگه
داشتن کل اجرا.

---

**دو دام در `.env` که راحت گرفتارشان می‌شوید:**

۱. **هر کلید فقط باید یک بار باشد.** `python-dotenv` آخرین مقدار را نگه
می‌دارد و هیچ حرفی هم نمی‌زند. پس اگر بالای فایل `MAX_STEPS=6` بگذارید ولی
`MAX_STEPS=12` قدیمی پایین‌تر مانده باشد، برنامه با ۱۲ اجرا می‌شود و هیچ
نشانه‌ای نمی‌بینید. پروژه خودش هشدار می‌دهد:

```
warning: 'MAX_STEPS' is set more than once in .env; only the LAST value is used
```

۲. **`.env` باید کنار `main.py` باشد**، نه داخل `src/`. و **`.env.example`
را ویرایش نکنید** — آن فقط یک الگوست و هرگز خوانده نمی‌شود. اگر تنظیماتتان
اعمال نمی‌شود، `python check_llm.py` را بزنید: خط اول خروجی می‌گوید کدام فایل
واقعاً خوانده شده.

### قبل از اجرا، قابلیت‌ها را بسنجید

```bash
python check_llm.py
```

سه فراخوانی کوچک می‌زند و می‌گوید مدل شما کدام قابلیت را دارد:

```
env file : C:\...\csv-insight-agent\.env
provider : openai
model    : gpt-6-luna
--------------------------------------------------------------
[  OK  ] build client
[  OK  ] plain chat  (reporter)
[  OK  ] structured output  (planner, supervisor)
[  OK  ] tool calling  (analyst, charter)
--------------------------------------------------------------
All three capabilities work. You are ready for a real run.
```

این گراف سه چیز متفاوت از مدل می‌خواهد و هر سه باید کار کنند:

| قابلیت | کدام نودها | اگر نباشد چه می‌شود |
|---|---|---|
| چت ساده | `reporter` | هیچ‌چیز کار نمی‌کند |
| structured output | `planner`, `supervisor` | مسیریابی ممکن نیست |
| **tool calling** | `analyst`, `charter` | ایجنت هیچ عددی درنمی‌آورد؛ گزارش خالی می‌شود |

برای سنجیدن یک مدل دیگر بدون دست‌زدن به `.env`:

```bash
python check_llm.py --provider anthropic
```

و برای اطمینان از این‌که خودِ `check_llm.py` سالم است:

```bash
python check_llm.py --provider fake      # باید هر سه OK بدهد
```

---

## قدم ۶ — اولین اجرای واقعی

```bash
python main.py data/sales.csv "Which region drives revenue, and is the trend going up?" --no-approval
```

`--no-approval` یعنی برای تأیید نقشه توقف نکن — برای اولین اجرا ساده‌تر است.
هزینهٔ تقریبی با `gpt-6-luna`: چند سنت. زمان: حدود ۳۰ تا ۹۰ ثانیه.

**اگر خطای کلید گرفتید**، چیزی شبیه این می‌بینید — و توجه کنید که برنامه
crash نمی‌کند، خطا را به‌عنوان داده ثبت می‌کند و تمیز تمام می‌شود:

```
  [loader] loaded 425 rows x 12 columns
  [planner] ! planner: OpenAIError: Missing credentials...
findings: 0   charts: 0
report:   not written
```

یعنی `.env` خوانده نشده یا کلید غلط است. چک کنید `.env` کنار `main.py` باشد.

**اگر `model not found` گرفتید:** نام مدل‌ها با زمان عوض می‌شود. لیست فعلی
provider خودتان را ببینید و `LLM_MODEL` را دستی درست کنید.

### گزارش را بخوانید — اینجا مهم‌ترین قضاوت شماست

```bash
cat outputs/report_*.md        # ویندوز: type outputs\report_*.md
```

**سه چیز را کنترل کنید:**

1. **عددها درست‌اند؟** یکی‌دو رقم را دستی چک کنید:
   ```bash
   python -c "import pandas as pd; d=pd.read_csv('data/sales.csv'); print(d.groupby('region')['revenue'].sum())"
   ```
   اگر عدد گزارش با این نخواند، مدل جایی از خودش درآورده — و آن یک باگ
   prompt است، نه باگ کد.

2. **بخش Caveats چیزی دارد؟** در `sales.csv` عمداً ۵ سطر تکراری، nullها، و
   یک سطر با درآمد صد برابر کار گذاشته‌ام. گزارش خوب حداقل یکی از این‌ها را
   می‌بیند. گزارشی که با اطمینان کامل و بدون هیچ caveat می‌نویسد، همان حالت
   خطرناکی است که باید یاد بگیرید تشخیص بدهید.

3. **نمودارها منطقی‌اند؟** فایل‌های PNG را باز کنید.

---

## قدم ۷ — تنظیم promptها

این قدم تمام نمی‌شود؛ کار اصلی ساختن ایجنت همین است.

اگر در اجرای واقعی یکی از این‌ها را دیدید، سراغ `src/prompts.py` بروید:

| نشانه | کدام prompt | چه کار کنید |
|---|---|---|
| finding ثبت نمی‌شود، ولی analyst حرف می‌زند | `ANALYST_SYSTEM` | قالب `FINDING:` / `EVIDENCE:` را با یک مثال کامل نشان بدهید |
| نمودار زیادی می‌سازد | `SUPERVISOR_SYSTEM` | سقف را صریح‌تر بنویسید |
| گزارش پرحرف و کلیشه‌ای است | `REPORTER_SYSTEM` | «بدون مقدمه، بدون نتیجه‌گیری» را تأکید کنید |
| ستون خیالی می‌سازد | `PLANNER_SYSTEM` | «فقط از نام ستون‌های پروفایل استفاده کن» را تکرار کنید |
| Caveats خالی است | `REPORTER_SYSTEM` | بخواهید nullها و سطرهای تکراری را صریحاً بررسی کند |

بعد از هر تغییر prompt، دو کار کنید:

```bash
pytest                    # مطمئن شوید چیزی نشکسته
python main.py data/sales.csv "همان سؤال قبلی" --no-approval
```

**و همان سؤال قبلی را بپرسید.** مقایسه فقط وقتی معنا دارد که سؤال ثابت
باشد. این ساده‌ترین شکل «eval» است، و تمرین شمارهٔ ۱۱ در `PRACTICE.md`
همین را جدی‌تر می‌کند.

---

## دادهٔ خودتان

وقتی روی `sales.csv` راضی شدید:

```bash
python main.py /path/to/your-data.csv "سؤال شما"
```

هر CSV با هدر ستون کار می‌کند. ستون‌هایی که اسمشان `date` / `time` /
`month` دارد خودکار به تاریخ تبدیل می‌شوند.

اگر فایلتان بزرگ است (بیش از چند صد هزار سطر)، `MAX_ROWS_TO_MODEL` را
پایین نگه دارید و `CODE_TIMEOUT` را ببرید بالا.

---

## قدم ۸ — اپ وب

اول پکیج تازه را نصب کنید (یک بار، با محیط مجازی فعال):

```powershell
pip install -r requirements.txt
```

بعد:

```powershell
python run_app.py
```

یا در File Explorer روی `run_app.bat` دوبار کلیک کنید. باید ببینید:

```
CSV Insight Agent 0.14.1  --  openai / gpt-5.6-terra
  on this computer:  http://localhost:8501
  Colleagues cannot connect: set APP_PASSWORD in .env to allow it.
```

و مرورگر باز می‌شود. اسمتان را بنویسید، **Sample data** را انتخاب کنید و یک
سؤال بپرسید. نقشه می‌آید، **Approve and run** را بزنید، و گزارش با نمودارها
روی صفحه ظاهر می‌شود. برای بستن اپ: `Ctrl+C` در همان پنجره.

### باز کردن اپ برای همکاران

1. در `.env` یک خط اضافه کنید (رمز دلخواه، حداقل ۸ حرف):

   ```
   APP_PASSWORD=یک-رمز-طولانی
   ```

2. `python run_app.py` را دوباره اجرا کنید. حالا یک خط دیگر چاپ می‌شود:

   ```
     for colleagues:    http://192.168.1.23:8501   (they need APP_PASSWORD)
   ```

3. ویندوز می‌پرسد Python اجازهٔ شبکه داشته باشد: فقط **Private networks** را
   تیک بزنید و Allow.
4. آدرس و رمز را به همکاران بدهید. تا وقتی پنجرهٔ `run_app` روی کامپیوتر شما
   باز است و کامپیوتر روشن است، اپ کار می‌کند.

اگر همکار صفحه را باز نمی‌کند: هر دو در یک شبکه هستید؟ پنجرهٔ `run_app` باز
است؟ فایروال را اجازه داده‌اید؟ (Windows Security → Firewall → Allow an app →
Python، تیک Private.)

**مهم:** هر کس رمز را دارد با کلید API شما کار می‌کند و از طریق مدل روی
کامپیوتر شما کد pandas اجرا می‌شود. فقط همکاران مورد اعتماد.

---

## نکتهٔ ویندوز

- **WSL لازم نیست.** timeout سندباکس از نسخهٔ 0.13 روی ویندوز هم کار می‌کند.

---

## چک‌لیست سریع

```
[ ] python --version  →  3.10+
[ ] محیط مجازی فعال (اسم .venv را در prompt ترمینال می‌بینید)
[ ] pytest  →  242 passed, 1 skipped (ویندوز)
[ ] --provider fake  →  گزارش + PNG در outputs/runs/<thread>/
[ ] توقف و ادامه با --resume کار می‌کند
[ ] .env با کلید واقعی، کنار main.py
[ ] اجرای واقعی  →  گزارشی که عددهایش را دستی تأیید کرده‌اید
[ ] python run_app.py  →  همان گزارش در مرورگر
```
