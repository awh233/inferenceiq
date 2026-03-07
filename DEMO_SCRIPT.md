# InferenceIQ 60-Second Demo Video Script

**Duration:** 60 seconds
**Target Audience:** Investors, Enterprise Customers, AI/ML Teams
**Style:** Professional, Fast-Paced, Data-Driven
**Tone:** Confident, Solution-Focused

---

## PRODUCTION NOTES

### Overall Aesthetic
- **Color Scheme:** Dark dashboard (navy/dark blue) with accent colors (cyan, green for savings)
- **Pacing:** Quick cuts every 3-4 seconds to maintain investor energy
- **Music:** Upbeat tech soundtrack, 120 BPM, minimal vocals (Epidemic Sound or similar)
- **Typography:** Clean sans-serif (Montserrat Bold for headers, Inter for body)
- **Transitions:** Fast cross-fades (200ms) or subtle slide transitions

---

## SCENE 1: HOOK (0-10 seconds)

### Visual Direction
- **Start:** Black screen with InferenceIQ logo fade-in
- **Transition:** Explosive zoom/pulse effect with sound sting
- **Main Visual:** Dashboard hero shot - full dashboard view showing live metrics
- **Key Elements on Screen:**
  - Large red banner: "Total Spend: $412,857"
  - Green highlight: "Total Savings: $156,034"
  - Live throughput graph animating
  - Multiple model cards (GPT-4, Claude 3.5, Gemini Pro)

### Voiceover Script
> **[TONE: Urgent, punchy, confident]**
>
> "Your AI bills are out of control.
> But what if they didn't have to be?"

### Key Metrics to Display
- "$156K saved"
- "38% cost reduction"
- "Real-time visibility"

### Duration: 10 seconds
- 0-2s: Logo animation + sting
- 2-10s: Dashboard hero shot with live metrics animating in

### Sound Design
- Sharp tech sting on logo reveal
- Subtle data visualization whoosh sounds as metrics appear
- Background music fades in under voiceover

---

## SCENE 2: PROBLEM (10-20 seconds)

### Visual Direction
- **Transition:** Split-screen effect - left side shows chaotic spending, right side shows InferenceIQ solution
- **Main Visuals:**
  - Red trending line going upward (AI inference spending)
  - Flickering question marks where costs should be
  - "?" icons representing unknown per-request costs
  - Gray out/reduced opacity on untethered vendor logos

### Voiceover Script
> **[TONE: Concerned, matter-of-fact, relatable]**
>
> "AI inference costs are growing 3-5X every year.
> Your teams are overpaying per request,
> and you have zero visibility into why."

### Supporting Visuals/Text Elements
- Animated upward arrow with "300% YoY Growth" label
- "Without Visibility" banner in red
- Stack of vendor logos (OpenAI, Anthropic, Google, Groq) appearing confused/unfocused
- Per-request cost estimates with "?" appearing randomly

### Statistics to Show
- "3-5X yearly growth in inference spending"
- "Zero per-request transparency"
- "$100K+ annually wasted on redundant requests"

### Duration: 10 seconds
- 10-12s: Problem setup visual
- 12-20s: Problem statistics and impact narrative

### Sound Design
- Descending tone or warning beep when showing growth
- Slight tension in music as problem is outlined
- Voice becomes slightly more urgent

---

## SCENE 3: SOLUTION DEMO (20-45 seconds)

### Part A: Dashboard Overview (20-28 seconds)

#### Visual Direction
- **Transition:** Wipe/reveal effect - problem fades away, solution dashboard appears
- **Main Visual:** Full InferenceIQ dashboard at inferenceiq.onrender.com
- **Animation:** Metrics appear with smooth number counting animations

#### Dashboard Elements to Highlight (in order):
1. **Top KPI Cards** (appears first):
   - Total Spend: $412,857
   - Total Savings: $156,034 (GREEN highlight)
   - Cost Reduction: 38%
   - Requests Processed: 47,382

2. **Real-time Throughput Chart** (animates):
   - Line chart showing requests over time
   - Color gradient from cyan to green
   - X-axis: Last 24 hours
   - Y-axis: Requests per minute

3. **Model Distribution Pie Chart**:
   - GPT-4 Turbo: 35%
   - Claude 3.5 Sonnet: 28%
   - Gemini Pro: 22%
   - Groq Mixtral: 15%
   - Each slice animates in with color flourish

#### Voiceover Script
> **[TONE: Confident, excited, specific]**
>
> "Meet InferenceIQ. Real-time AI cost optimization,
> from a single dashboard."

#### Key Numbers to Feature
- $156,034 saved
- 47,382 requests optimized
- 38% cost reduction
- 24-hour visibility

#### Duration: 8 seconds
- 20-24s: KPI cards appear with counting animation
- 24-28s: Charts animate into view

#### Sound Design
- Uplifting music tempo increases
- Subtle "data ping" sounds as each metric appears
- Positive confirmation tone when savings are highlighted

---

### Part B: SDK Implementation (28-35 seconds)

#### Visual Direction
- **Transition:** Dashboard zoom out, code editor zoom in (diagonal wipe)
- **Main Visual:** VS Code or IDE showing code before/after comparison
- **Layout:** Split screen (BEFORE on left, AFTER on right)

#### Code Block 1 (BEFORE):
```python
# Your existing code (3-4 lines)
from openai import OpenAI

client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}]
)
```

#### Code Block 2 (AFTER):
```python
# With InferenceIQ (2 lines added)
from inferenceiq import InferenceIQ

iq = InferenceIQ(api_key="iq-...")
response = iq.chat.completions.create(  # Drop-in replacement
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}]
)
```

#### Visual Highlights:
- Green highlight box around the 2-line addition
- Arrow pointing to additions: "2-Line Setup"
- Checkmark appears next to optimized version
- Real-time savings counter appears: "+$47 saved on this request"

#### Voiceover Script
> **[TONE: Developer-focused, simple, empowering]**
>
> "Two lines of code.
> Plug in InferenceIQ.
> Watch your costs plummet."

#### Key Callouts:
- "pip install inferenceiq"
- "Drop-in replacement for OpenAI, Anthropic, Google, Groq"
- "Immediate savings"

#### Duration: 7 seconds
- 28-30s: Code editor appears, BEFORE code visible
- 30-33s: AFTER code appears with highlights
- 33-35s: Savings counter animates, checkmark

#### Sound Design
- Keyboard typing sounds (subtle, not overdone)
- Positive "success" chime when optimization is applied
- Music maintains upbeat tempo

---

### Part C: Real-Time Savings Ledger (35-42 seconds)

#### Visual Direction
- **Transition:** Code editor zooms out, dashboard returns with new view
- **Main Visual:** Savings Ledger table/list view showing per-request transparency
- **Animation:** Rows appear in real-time as requests are processed
- **Color Coding:**
  - Green for successful optimizations
  - Light blue for cached responses
  - Yellow for model swaps

#### Savings Ledger Columns (left to right):
| Timestamp | Original Cost | Optimized Cost | Savings | Method | Model |
|-----------|---------------|----------------|---------|--------|-------|
| 14:32:15 | $0.08 | $0.048 | $0.032 (40%) | Cache Hit | GPT-4 |
| 14:31:58 | $0.12 | $0.036 | $0.084 (70%) | Model Routing | Claude 3.5 |
| 14:31:42 | $0.05 | $0.05 | $0 (0%) | No Optimization | GPT-4 |
| 14:31:31 | $0.10 | $0.03 | $0.07 (70%) | Semantic Cache | Gemini |
| 14:31:15 | $0.15 | $0.045 | $0.105 (70%) | Model Routing | Groq |

#### Visual Highlights:
- Real-time row additions (new requests appearing live)
- Total savings counter at bottom: "Total Saved Today: $1,847.32"
- Green progress bar showing cost reduction % accumulating
- Hover effect showing optimization method details

#### Voiceover Script
> **[TONE: Matter-of-fact, reassuring, analytical]**
>
> "See exactly where your savings come from.
> Every request. Every dollar.
> Full transparency."

#### Key Metrics to Display:
- Individual request savings ($0.03-$0.10 per request)
- Optimization methods: "Semantic Caching", "Model Routing", "Cache Hit"
- Cumulative daily savings: "$1,847.32"
- Average optimization rate: "62% cost reduction"

#### Duration: 7 seconds
- 35-38s: Ledger table appears with column headers
- 38-42s: Live rows stream in with savings accumulating

#### Sound Design
- Gentle "cash register" or "positive impact" sound as each row appears
- Music continues upbeat but becomes more grounded/trustworthy tone
- Voiceover slower, more deliberate pacing (building confidence)

---

### Part D: Model Routing In Action (42-45 seconds)

#### Visual Direction
- **Transition:** Ledger zooms out, animated model routing flow appears center-screen
- **Main Visual:** Interactive model selection flowchart
- **Animation:** Request flows from center, branches to different models based on optimization rules

#### Visual Flow:
```
[Incoming Request]
    ↓
[InferenceIQ Optimizer]
    ├→ [Check Cache] → Hit? → ✓ Return Cached Response (70% savings)
    ├→ [Semantic Similarity] → Cached? → ✓ Use Similar Response (40% savings)
    ├→ [Model Eval] → Can Claude 3.5 handle it? → ✓ Route to Claude (60% savings)
    ├→ [Analyze Complexity] → Need GPT-4? → ✓ Route to GPT-4 (essential tasks)
    └→ [Cost Analysis] → Groq fast enough? → ✓ Route to Groq (80% savings)
```

#### Visual Elements:
- Central hub with "InferenceIQ" label
- 5 colored pathways branching to model icons:
  - Cyan path → Claude 3.5 Sonnet (blue icon)
  - Green path → Groq Mixtral (green icon)
  - Purple path → Gemini Pro (orange icon)
  - Red path → GPT-4 Turbo (red icon)
  - Gold path → Cache Return (gold glow)
- Animated request "balls" flowing through optimal paths
- Real-time decision labels: "90% Quality Match", "47% Cost Reduction", "Identical Context"

#### Voiceover Script
> **[TONE: Technical confidence, forward-thinking]**
>
> "Intelligent routing. Semantic caching.
> The right model, at the right price, every time."

#### Key Features to Highlight:
- "Supports OpenAI, Anthropic, Google, Groq, DeepSeek"
- "Intelligent model selection"
- "Semantic request caching"
- "Cost-quality optimization"

#### Supporting Text:
- "40-95% cost reduction per request"
- "Zero latency addition"
- "Quality score: 99.2% maintained"

#### Duration: 3 seconds
- 42-45s: Model routing diagram animates, decisions flow through system

#### Sound Design
- Futuristic "routing" sounds (subtle sci-fi effects)
- Data flowing sounds
- Music builds to crescendo as call-to-action approaches

---

## SCENE 4: CALL TO ACTION (45-60 seconds)

### Visual Direction
- **Transition:** Model routing diagram zooms out and fades
- **Main Visual:** Clean, bright end card with three clear CTAs arranged horizontally
- **Background:** Subtle gradient (dark blue to navy), animated particles or data flow pattern
- **Layout:** Three equal sections, each highlighting one action

#### CTA Section 1: Python SDK
- **Icon:** Terminal/Code icon
- **Headline:** "pip install inferenceiq"
- **Subheader:** "Drop-in replacement for your AI API calls"
- **Visual:** Terminal window showing command
- **QR Code:** Points to PyPI package page (optional)
- **Color Accent:** Cyan/bright blue

#### CTA Section 2: Live Dashboard
- **Icon:** Dashboard/Charts icon
- **Headline:** "inferenceiq.onrender.com"
- **Subheader:** "See real-time savings"
- **Visual:** Mini dashboard preview thumbnail
- **QR Code:** Points to live dashboard (optional)
- **Color Accent:** Green

#### CTA Section 3: Book a Demo
- **Icon:** Calendar/Meeting icon
- **Headline:** "Book a Demo"
- **Subheader:** "Let us optimize your specific use case"
- **Visual:** Calendar interface or contact form preview
- **QR Code:** Points to booking page (optional)
- **Color Accent:** Orange/warm accent

### Voiceover Script (Part 1: 45-55 seconds)
> **[TONE: Direct, powerful, closing the loop]**
>
> "Ready to cut your AI costs by up to 95%?
>
> Install InferenceIQ today.
> Visit our live dashboard to see what your savings could be.
> Or book a demo to optimize your specific workload."

### Voiceover Script (Part 2: 55-60 seconds)
> **[TONE: Confident, memorable, final]**
>
> "InferenceIQ: Intelligent AI Cost Optimization.
>
> Make every inference count."

### End Card Layout

```
┌─────────────────────────────────────────────────────────┐
│                                                           │
│  ┌──────────────┬──────────────┬──────────────┐         │
│  │   📦 CODE    │   📊 LIVE    │   📅 DEMO    │         │
│  │              │   DASHBOARD  │              │         │
│  │ pip install  │ inferenceiq. │ Book a Demo  │         │
│  │ inferenceiq  │ onrender.com │              │         │
│  │              │              │ 40-95% Cost  │         │
│  │ Drop-in API  │ Real-time    │ Reduction    │         │
│  │ Replacement  │ Savings      │              │         │
│  └──────────────┴──────────────┴──────────────┘         │
│                                                           │
│              InferenceIQ.com                             │
│         Intelligent AI Cost Optimization                │
│                                                           │
│              "Make every inference count"                │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Visual Elements
- **Logo:** InferenceIQ logo prominently at bottom center
- **Tagline:** "Make every inference count" in white text, elegant serif font
- **Colors:** Maintain dark professional aesthetic with vibrant accent colors
- **Animation:** Subtle pulse on each CTA section as voiceover mentions it
- **Border:** Thin cyan or gold border around each CTA option
- **Contact Info:** Email (support@inferenceiq.com) or website URL at bottom

### Duration: 15 seconds
- 45-47s: End card fades in with CTAs appearing sequentially
- 47-55s: Voiceover describes three options while each section highlights
- 55-60s: Final tagline, logo emphasis, fade to black

### Sound Design
- Music reaches peak during CTA presentation
- Clear, confident tone in voiceover
- Subtle button "focus" sounds as each CTA is mentioned
- Music fades smoothly as video ends
- Optional: 2-second silence before final fade to black for impact

---

## TECHNICAL PRODUCTION CHECKLIST

### Dashboard Recordings
- [ ] Record full dashboard at inferenceiq.onrender.com (1080p, 60fps)
- [ ] Ensure live metrics are visible and animating
- [ ] Capture real savings data (not mock data)
- [ ] Record throughput chart with 24-hour history
- [ ] Capture model distribution pie chart
- [ ] Record savings ledger with live row streaming
- [ ] Capture model routing diagram animation

### SDK Demo
- [ ] Use clean VS Code with default theme
- [ ] Record in 1080p, 60fps
- [ ] Show syntax highlighting clearly
- [ ] Use Menlo or Monaco font at 16pt minimum
- [ ] Include real import statements and API keys (obfuscated for security)
- [ ] Show actual API call execution

### Graphics & Motion
- [ ] Logo animation (2-3 seconds): fade in with scale/pulse
- [ ] Transition effects: cross-fade (200ms), wipe, or slide
- [ ] Chart animations: line drawing, pie slice expansion
- [ ] Number counters: animate from 0 to final value (1-2 seconds)
- [ ] Text reveals: staggered or cascading

### Color Palette
- **Primary Dark:** #0f172a (navy background)
- **Primary Dark (Lighter):** #1a2140
- **Accent Green (Savings):** #10b981 or #22c55e
- **Accent Cyan:** #06b6d4 or #0ea5e9
- **Accent Orange:** #f97316 or #fb923c
- **Text White:** #ffffff
- **Text Gray:** #94a3b8
- **Error Red:** #ef4444 or #dc2626

### Font Stack
- **Headers:** Montserrat Bold, 28-32px
- **Subheaders:** Montserrat SemiBold, 18-22px
- **Body Text:** Inter Regular, 14-16px
- **Monospace (Code):** Monaco or Menlo, 14-16px

### Music & Sound
- **Background Music:** 120 BPM, tech/corporate upbeat
- **Duration:** 60 seconds (loop or fade out)
- **Style:** Instrumental, minimal vocals, professional
- **Suggested Sources:**
  - Epidemic Sound
  - Artlist
  - AudioJungle
  - Soundly
- **Additional Sound Effects:**
  - Logo sting (1-2 seconds, tech chime)
  - Data visualization whoosh (0.5 seconds, x3-4)
  - Positive confirmation tone (0.3 seconds, x5-6)
  - Keyboard typing (subtle, looped 2-3 seconds)
  - Cash register/savings notification (0.5 seconds, x5-7)
  - Button focus sounds (0.2 seconds, x3)

### Video Specifications
- **Resolution:** 1920x1080 (16:9)
- **Frame Rate:** 60fps
- **Codec:** H.264 (MP4) or ProRes (professional edit)
- **Bitrate:** 15-20 Mbps (H.264), variable (ProRes)
- **Color Space:** Rec. 709 (sRGB)
- **Audio:** Stereo, 48kHz, -3dB normalization

---

## DIRECTOR'S NOTES

### Key Messaging Priorities
1. **Immediate Cost Savings:** Lead with specific numbers ($156K saved, 38% reduction)
2. **Ease of Integration:** Emphasize "2-line setup" and "drop-in replacement"
3. **Transparency:** Highlight per-request cost visibility
4. **Investor Appeal:** Show scale (47,382 requests processed), ROI (38% cost reduction), and vendor diversity
5. **Technical Credibility:** Display real data, actual dashboard metrics, functional SDK demo

### Pacing Strategy
- **Scenes 1-2 (0-20s):** Rapid, energetic (hook investor attention, establish problem)
- **Scene 3 (20-45s):** Steady, detailed (demonstrate solution with proof points)
- **Scene 4 (45-60s):** Deliberate, confident (clear CTA, memorable tagline)

### Camera/Recording Tips
- Use smooth zooms and pans (no jittery mouse movements)
- Keep dashboard visible for 5+ seconds on each view to allow reading
- Use cursor highlights or subtle pointer animations for emphasis
- Avoid cluttered backgrounds; keep focus on product UI
- Record multiple takes of each section for editing flexibility

### Accessibility Considerations
- Ensure all text is large enough and high contrast (WCAG AA compliant)
- Include captions/subtitles for voiceover in final video
- Use color combinations that work for colorblind viewers
- Avoid flashing or rapid transitions (max 3 flashes per second)

### Editing Timeline
- Scene 1 Hook: 10 seconds
- Scene 2 Problem: 10 seconds
- Scene 3a Dashboard: 8 seconds
- Scene 3b SDK: 7 seconds
- Scene 3c Ledger: 7 seconds
- Scene 3d Routing: 3 seconds
- Scene 4 CTA: 15 seconds
- **Total: 60 seconds**

### Post-Production Workflow
1. Color grade all dashboard footage (boost saturation slightly, increase contrast)
2. Apply noise reduction and audio leveling
3. Add motion graphics (counters, pie charts, routing diagrams)
4. Layer voiceover and adjust levels (-3dB for music)
5. Add sound effects (timing critical for engagement)
6. Export master file, create social media variants (15s, 30s, 45s cuts)
7. Add captions/subtitles in post
8. Review with stakeholders and iterate

### Distribution Format Recommendations
- **Primary:** YouTube (16:9, 1080p, H.264)
- **Social Media:** TikTok/Instagram Reels (9:16 vertical version)
- **Website:** Auto-play on inferenceiq.com homepage (muted, looped)
- **LinkedIn:** Native upload (1080p, captions burned in)
- **Investor Decks:** Embedded in PowerPoint/Google Slides (720p, looped)
- **Email Campaigns:** Link to YouTube (not embedded video)

---

## APPROVAL SIGN-OFF

**Draft Status:** Ready for Production
**Last Updated:** 2026-03-07
**Estimated Production Time:** 3-5 days (recording, editing, revisions)
**Budget Estimate:** $2,000-$5,000 (videographer, editor, music license)

**Next Steps:**
1. Schedule recording session with product team
2. Prepare live dashboard with optimal metrics
3. Prepare SDK demo environment
4. Source background music and sound effects
5. Create motion graphics assets
6. Record and edit
7. Review and iterate
8. Publish and distribute

---

## APPENDIX: KEY PRODUCT STATS TO FEATURE

- **Cost Reduction:** 40-95% per request optimization
- **Current Dashboard Stats:** $156,034 saved, 38% cost reduction, 47,382 requests processed
- **Supported Providers:** OpenAI, Anthropic, Google, Groq, DeepSeek
- **API Endpoints:** 12 REST API endpoints for comprehensive integration
- **Setup Time:** 2 lines of code (drop-in replacement)
- **Infrastructure:** Live dashboard at inferenceiq.onrender.com, live API at inferenceiq-api.onrender.com
- **Package:** Python SDK via pip (pip install inferenceiq)
- **Performance:** Typically 0-5ms added latency, maintains 99%+ quality
- **Key Features:** Semantic caching, intelligent model routing, real-time cost transparency, per-request optimization

---

**Video Ready for Production. All elements specified. Ready to shoot.**
