# Product & business adaptation (S2.T3.1)

Written artifact for the **~20%** Product & Business rubric: concrete UX, deployment, and parent-facing design for low bandwidth, shared devices, multilingual use, and non-literate families.

---

## 1. First 90 seconds of UX (6-year-old, Kinyarwanda-dominant speaker)

**0:00–0:10 — Greeting, no reading required**  
The child sees one full-screen **Start** (large touch target, well above 48 dp) and a **sound** icon. The **first** audio is in **Kinyarwanda**: e.g. *“Muraho! Twige imibare.”* (Hello! Let’s do a little math.) French and English are optional, **icon-based** chips, not a wall of text. No sign-up, no keyboard on first launch.

**If the child stays silent for 10 seconds** (has not tapped Start): play a **shorter** Kinyarwanda line (*“Kanda aha, tugende.”* — tap here, let’s go) and **pulse** the Start control once (subtle scale or border) so a non-reader can follow. No error text that requires reading.

**0:10–0:50 — First task: one clear win**  
The first task is **tap-only**: a simple count-the-picture item (2–3 objects) with **large** number buttons, **haptic** on press, **green** animation on success. A **Repeat** control replays the **spoken** question. **Typed answer and microphone** stay **off** on the first run to cut load and ASR errors.

**0:50–1:30 — Reward, then one small step**  
Short Kinyarwanda praise (TTS), then **one** follow-up in the same tap style. **Help (?)** highlights the **objects** in the image (visual grounding) without giving away the count. **Pause** is one control, top corner, one tap.

**Success for the 90s window (design target):** after **Start**, at least one correct try with only a **few** taps, **no required reading** of Latin script, and a defined response to **10 s silence** (second line + button pulse).

---

## 2. Shared tablet for 3 children at a community centre

**Context**  
One **low-cost Android** tablet, **offline-first** (no child accounts in the cloud in the default design). Power and connectivity are **intermittent**.

**How learners are separated**  
At launch, a **picture grid** (e.g. 6 animals or colours) lets each child pick **their** face without typing. Internally, sessions use stable ids such as `child-a`, `child-b`, `child-c` (or nicknames the facilitator sets once). Progress rows in SQLite are keyed by **HMAC(learner_id)** with a device secret (`TUTOR_HMAC_SECRET`) so file inspection does not expose raw names if that matters for your setting.

**Switching between 3 children**  
**“Next child”** clears the current question, stops audio, and returns to the **profile grid** in two taps. The next child does not see the previous child’s last screen or answer.

**Privacy**  
- Data stays on the device until a **facilitator** runs an export or sync.  
- No photos of children in the default export path.  
- Differential-privacy **noise** on shared statistics is documented for **teachers**, not on the child screen.

**Reboot or power loss**  
The database file is a **single** local SQLite store. After reboot, the app opens on the **profile grid**; children re-select their icon. If the file is **corrupt**, the app offers a **fresh local session** for play and a **teacher-only** path to restore from backup (USB), not a blocking error for the child.

**Fair use in a 45-minute block**  
Optional soft cap (e.g. **~12–15 minutes active** per child) with a **break** full-screen, to support turn-taking in a class of three with one device.

---

## 3. Non-literate parent report (design for “read in ~60 seconds”)

**Principle**  
The parent may not read English or French. The report must be **scannable by icons and one big number**, with **optional** Kinyarwanda audio in **~45–60 seconds** (not a long podcast).

**One page (A5 or phone portrait), top to bottom**

1. **Child** — Avatar or photo + first name (or avatar only if names are sensitive). **No** internal learner_id on the page.

2. **This week in five “batteries”** — One row of **5 icons** (counting, number sense, add, subtract, word problems). Each icon fills like a **battery** 0–100% so “good week” is visible without sentences.

3. **Big number** — One two-digit (or one-digit) **largest number practised** this week (e.g. **7**) as a simple progress anchor for families who care about “bigger numbers.”

4. **Time or streak (optional, pictorial)** — A **clock** + “12” = 12 minutes this week, or **flame** icons for days in a row, without dense text.

5. **QR code** — Points to a **short** Kinyarwanda audio (hosted with consent): e.g. *“Yige neza imibare; agakomeza.”* (They practised well; they’re making progress.) **No** login to listen.

6. **Lock icon** — One line the **facilitator** can read aloud: data stayed on the tablet; nothing sold to advertisers.

**Mapping to this repo**  
Aggregates from `parent_report.py` and `parent_report_schema.json` feed the icon row and the big number. Teacher-facing notes on **ε** for DP stay off this sheet.

**Offline parent**  
If there is no phone, the facilitator gives a **pre-printed** template and ticks or shades the batteries by hand; the layout still “reads” in a minute by shape and colour.

---

*End of product & business artifact.*
