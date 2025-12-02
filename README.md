# Taskwarrior Nautical ⚓︎⛓

**Chains and Anchors for Taskwarrior** -- The missing layer between what your task manager can do and what your actual life requires.

While other tools force you to choose between oversimplified repeats and cloud-based complexity, Nautical gives you enterprise-grade scheduling with local privacy. Express any real-world pattern in plain English, then watch it execute with mathematical precision.

Have you ever found yourself thinking:

- **"This should happen on the 2nd Monday, but only in April and June."**
    
- **"I need this every 33 hours exactly, starting from when I actually complete it, not some arbitrary schedule."**
    
- **"Why can't I just say 'first and last day of the month' and have it work?"**
    

You're not alone. I hit those exact walls with traditional task management. As someone who values precision and reliability, I needed tools that could handle real-world complexity without becoming a burden themselves.

Nautical gives you the expressive power to describe any real-world rhythm in plain English--from "2nd Monday with roll-forward" to "random weekday in the first half of the month"--then executes it with mathematical precision, keeping your data local and your privacy intact.

**Here's what that means in practice:**

Instead of wrestling with basic repeats that create more work than they save, or surrendering your data to yet another cloud service, you get:

```
# Business logic that understands reality
task add "Monthly review" anchor:"m:-1"
```
![lastday](https://github.com/user-attachments/assets/95a35bd9-3ee9-4e89-9b73-dfb8588af73e)
```
# Complex patterns made simple  
task add "Team planning" anchor:"w:mon + m:1:15"
```
![complex](https://github.com/user-attachments/assets/00694fad-3085-45f2-a9aa-8a7ca402fb1d)
```
# Exact timing without drift
task add "Medication" cp:8h chain:on due:today+14h
```
![simple chain](https://github.com/user-attachments/assets/3b33003e-71ed-4e5f-8915-aa8e0b3a8fe1)

### What Makes Nautical Different

Nautical isn't trying to be everything to everyone. It solves one problem exceptionally well: **helping you set and forget repeating tasks, trusting they'll appear exactly when they should, in the exact right quantity.**

**Intentional Scheduling** - Express schedules that match reality, not just basic repeats:

- `m:last-fri` → Last Friday of the month.

- `w:mon,wed + m:1:15` → Mondays and Wednesdays that fall in the first half of the month.

- `m:1@next-mon` → First of the month if Monday or roll to the next Monday.

- `w:mon@t=09:00,wed@t=15:00` → Monday at 09:00, Wednesday at 15:00.

- `y:05:15@prev-sat` → 15 of May if Saturday or roll to the previous Saturday.

- `w:rand` →  A random day per week.

- `cp:8d`  →  8 days after completion.

**⚡ Mathematical Precision**

- No floating point time calculations

- No daylight saving time surprises

- No creeping drift over months/years

- UTC math with local time display
---


## Nautical Manual

This readme is an introduction to the system, for detailed information please check the TW-Nautical-Manual.pdf

The Manual includes:

* A library of copy‑ready commands for common patterns.
* A detailed description of the pattern language.
* Business‑day rules, bucket ranges, seeded random.
* Anything else you might want to know about Nautical.
---

## Part I - Chains - Classic Chained Recurrence

Need the next occurrence to be **based on when you complete** the current one?

Want to **keep the same due time** for day‑based periods (e.g., always 09:00) even though you complete early or late?

Run odd cadences like **every 28 hours** and want **exact adds with no drift**?

### Real‑world examples

**1) Trim the garden grass every 12 days at 09:00**
```
task add "Trim the grass" due:tomorrow+9h cp:12d
```
![12d](https://github.com/user-attachments/assets/deb5592d-b756-44be-ad88-1222bd5d06d3)

The next link is scheduled **12 days later**; because 12 days is a multiple of 24h, Nautical **preserves the 09:00 due time** so your routine stays fixed.


**2) Take a vitamin every 36 hours**
```
task add "Take the vitamin" due:today+15h cp:36h
```
The next link is **exactly end (completion time) + 36h**.
![36h](https://github.com/user-attachments/assets/20b9a106-f94f-41d8-be50-8eda18750b7a)


**3) Chain task with a max cap
```
task add "Tool calibration" cp:P3D chainMax:5 due:today+12h
```
What happens: panels show **links left** and the **final date**. When you reach the last link, Nautical stops  -  no extra spawns.
![3d](https://github.com/user-attachments/assets/b4363406-aecf-4754-9a62-cf8ff6b79430)


## Part II - Anchors - Real-World Patterns for Sophisticated Workflows

**What it is**
A compact pattern language to express real calendar logic  -  weekly, monthly, or yearly anchors  -  with business‑day rolls, explicit times, AND/OR operators and parentheses.

### Is this you for you?

Need a task that recurs only on **Monday and Friday**? Or Monday at 11:00 and Friday at 18:00?

Want **1st and last day** of every month in one line?

Need **2nd Monday** or **last Friday** of the month?

Need the **first business day** when the 1st is a weekend?

Need the **nearest weekday**, or the **previous/next business day**?

Want **Mondays that are also the 1st or 15th** (AND), or **either 1st Saturday or 3rd Friday** (OR)?

Want **one random weekday each month** ?

If you answered **yes** to any of the above, Nautical can provide you with **even more**.

### Anchor examples

**Last Friday each month**

```
task add "Do the monthly review" anchor:m:last-fri due:today
```

**Nearest weekday to the 15th**

```
task add "Mid‑month meeting" anchor:m:15@nw anchor_mode:all due:today
```

**5th business day**

```
task add "Parcel delivery" anchor:m:5bd due:today
```

**Previous business day before the 1st**

```
task add "Month‑end report" anchor:m:1@pbd anchor_mode:all due:today
```

**Next Monday after the 1st**

```
task add "Monthly planning" anchor:m:1@next-mon due:today
```

**Bucket + random** - one random day among 1–7 each month (weekdays only)

```
task add "Focus day (early bucket)" anchor:'m:1:7 + m:rand@bd' due:today
```

**Quarterly review**  -  simple yearly list

```
task add "Quarterly review" anchor:y:01-15,04-15,07-15,10-15 anchor_mode:all due:today
```

**Leap day**  -  every Feb 29 (leap years only)

```
task add "Leap‑day check" anchor:y:29-02 due:today
```

> **Many more in the Nautical Manual:** buckets (e.g., `m:22:28`), nth‑weekday (`m:3wed`), year‑month random (`y:rand-10`), and combining patterns with parentheses/operators.

**Sourdough bake day**  -  first Saturday each month

```
task add "Sourdough bake day" anchor:m:1sat due:today
```

**3rd Wednesday of the month**

```
task add "Parent–teacher night" anchor:m:3wed anchor_mode:all due:today
```


###  **Three Anchor Modes for Different Work Types**

- **Skip** (default): Missed it? Move on. Perfect for "nice to have" tasks.

- **All**: Backfill everything. Essential for compliance and critical work.

- **Flex**: Skip past dates but respect anchors going forward. The balanced choice. Check the manual for when to use this mode.


---

## What carries forward to the next link

When a new link is spawned, Nautical keeps your context:

* **Standard fields:** description, project, tags, priority (when present).
* **Chain metadata:** `chain:on`, incremented `link`, and short‑UUID links via `prevLink`/`nextLink`.
* **All UDAs** from the parent (e.g., `area`, `value`, or any custom flags you use).
* **Dependencies** pointing to other tasks.
* **Annotations** with their original timestamps (Nautical uses a safe export→import so history remains true).

---

## Panels that explain themselves

Every add and complete shows a short, readable panel:

* **Pattern** and **Natural** language summary.
* **Basis**  -  how the next date was chosen (after due / after end / keep due time / exact add).
* **Next Due** with a delta (e.g., “in 2d 5h”).
* **Links left** and **Final (max/until)** when capped.
* **Timeline** with the last three and next three links (short UUIDs, clear “last link” marking).
* **Finish summary** when a chain ends (counts, best/worst/average timing, span).

---

# Install

```
# 1. Get the hooks
cd ~/.task/hooks
wget https://github.com/catanadj/taskwarrior-nautical/raw/main/on-modify-nautical.py
wget https://github.com/catanadj/taskwarrior-nautical/raw/main/on-add-nautical.py
chmod +x on-*-nautical.py
cd ..
wget https://github.com/catanadj/taskwarrior-nautical/raw/main/on-modify-nautical.py


# 2. Add to your ~/.taskrc
echo "
#
#░█▀█░█▀█░█░█░▀█▀░▀█▀░█▀▀░█▀█░█░░  
#░█░█░█▀█░█░█░░█░░░█░░█░░░█▀█░█░░  
#░▀░▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀
## Classic Chain Recurrence
uda.cp.type=duration 
uda.cp.label=Chain Period
uda.chain.type=string
uda.chain.label=Chain Status
uda.chain.values=on,off
uda.chain.default=off

## Advanced Anchor Recurrence 
uda.anchor.type=string
uda.anchor.label=Anchor
uda.anchor_mode.type=string
uda.anchor_mode.values=flex,all,skip
uda.anchor_mode.default=skip
uda.anchor_mode.label=Anchor Mode

## Limits
uda.chainMax.type=numeric
uda.chainMax.label=Chain Max
uda.chainUntil.type=date
uda.chainUntil.label=Chain Until

## Lineage
uda.prevLink.type=string # in Taskwarrior 3.4.2+ you can change this to type=UUID
uda.prevLink.label=Previous Link
uda.nextLink.type=string # in Taskwarrior 3.4.2+ you can change this to type=UUID
uda.nextLink.label=Next Link
uda.link.type=numeric
uda.link.label=Link Number
#
" >> ~/.taskrc

# 3. Install Rich if you don't have it already
pip install rich

# 4. Set the NAUTICAL_TZ env variable to your own timezone, the default is set to Australia/Sydney. You can also set this by modifying line 19 in nautical_core.py
export NAUTICAL_TZ=Continent/City # run for current session only; add it to your .bashrc/.zshrc file for permanence.
 
# 5. Test with a sophisticated pattern
task add "System test: 2nd Monday of the month" anchor:"m:2mon" due:today


```

---
## Determinism & performance

* **Seeded random.** `rand` picks are stable for each chain (for calendar anchor picks and panel colours).
* **Designed to be fast on Termux.** The core and hooks cache common lookups and avoid heavy work on the hot path.
* **Local only.** No network; everything stays on your machine.  Your data, your rules, your control.

---

## Requirements

* Taskwarrior 2.6+
* Python 3.9+
* `rich` for pretty panels


---

## Bug reporting & Development

**If you encounter unexpected behavior:**

- Check the Nautical Manual
- Verify your UDA configuration matches the expected setup
- Review the panel output for clues about what Nautical computed

**When something breaks:** Open a GitHub issue with:

- Your pattern or chain configuration
- The unexpected behavior (what you expected vs. what happened)
- Relevant panel output or task details
- Your Taskwarrior version and environment (desktop/Termux/etc.)

The more context you provide, the faster I can isolate and fix the issue.

If you need a new anchor pattern supported, something can be more intuitive or just having an idea on how to make this better please let me know. 

---
## Support

If you find this tool helpful, any support will be greatly apreciated.

You can do so [here](https://buymeacoffee.com/catanadj). Thank you.


---
## Why This Matters
Most productivity tools treat recurring tasks as an afterthought. They give you basic repeats that create more work than they save, or they're so complex that you spend more time managing the system than doing the work.

With Taskwarrior and Nautical, You can stop thinking about scheduling and start thinking about what matters -- doing.

That's the real promise of Nautical. It's not just about managing tasks - it's about reclaiming mental space. It's about having a system you can trust so completely that you can focus on the work that actually matters.

Your brain is for solving problems, not remembering schedules. Your tools should handle the predictable so you can focus on the meaningful.

TaskWarrior for the Real World -- because your work deserves tools that understand reality.

Deus vult.
