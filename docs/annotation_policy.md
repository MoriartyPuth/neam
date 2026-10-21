# News annotation policy v1

This is the project-specific flat PER/ORG/LOC policy. It is not a claim that all
external datasets use the same span boundaries. Store source text unchanged and
use exclusive-end Python character offsets. Do not infer entities from a keyword.

- PER: a person's complete name, including a directly attached short
  conventional honorific such as លោក or អ្នកស្រី. Exclude ceremonial titles
  such as សម្តេចមហាបវរធិបតី and ឯកឧត្តម, job descriptions, and military ranks.
  Tag លោក ហ៊ុន ម៉ាណែត together, but tag only ហ៊ុន ម៉ាណែត after a ceremonial
  title. A role without a name, such as មន្ត្រីសន្តិសុខ, is O.
- ORG: a named company, institution, government body, media organization, or
  organized group. Include its directly attached entity designator, for example
  ក្រុមហ៊ុនឯកជន VistaJet. Generic ក្រុមហ៊ុន, ឯកជន, រោងចក្រ, អង្គការ,
  and អាជ្ញាធរមូលដ្ឋាន are O unless they form a named institutional mention.
  Earlier curricula treating generic institutional classes as ORG are legacy
  regressions and must be normalized before reuse under this policy.
  A social-media page title in a photo credit is ORG only when the page is an
  identifiable organization or publisher; otherwise it is O. A city name inside
  a page title is LOC only when the mention refers to the geographical place.
- LOC: countries, cities, named regions and physical facilities. Include directly
  attached geographical designators, such as ប្រទេស or ព្រលានយន្តហោះ. A named
  facility is LOC; its operating company is ORG. In មន្ត្រីសន្តិសុខអាមេរិក,
  annotate អាមេរិក as LOC and the role as O. Do not label a generic facility.
- Nationality and people-group phrases are O. In ជនជាតិឥណ្ឌូនេស៊ី ("Indonesian
  nationals") and ជនជាតិខ្មែរ, neither ជនជាតិ nor the country or people name is
  an entity, because the phrase describes people, not the place. The same name
  used as a place, as in ពីកម្ពុជា or នៅប្រទេសឥណ្ឌូនេស៊ី, is LOC. A country naming
  the state that an official or body belongs to keeps the rule above (អាមេរិក in
  មន្ត្រីសន្តិសុខអាមេរិក is LOC). Inside a named organization it stays part of the
  ORG span (ស្ថានទូតឥណ្ឌូណេស៊ី).
- Only the fixed ជនជាតិ phrase is a nationality. A country after ប្រជាជន,
  ពលរដ្ឋ, ដៃគូ or a group noun such as អ្នកស្ម័គ្រចិត្ត names the country's
  population or partners and stays LOC (កម្ពុជា in ប្រជាជនកម្ពុជា and
  អូស្ត្រាលី in អ្នកស្ម័គ្រចិត្តអូស្ត្រាលី).
- A government named with its country is one ORG: រាជរដ្ឋាភិបាលកម្ពុជា and
  រដ្ឋាភិបាលអូស្ត្រាលី. A bare រាជរដ្ឋាភិបាល without a country is O. The same
  applies to a court, prosecution office or prison named with its province
  (អយ្យការអមសាលាដំបូងខេត្តព្រះសីហនុ, ពន្ធនាគារខេត្តសៀមរាប).
- Keep the project's existing full facility spans for compatibility, including
  descriptive prefixes immediately introducing a facility name. Do not extend
  the span into following verbs, conjunctions, or unnamed references.
- Do not nest a country LOC inside a larger ORG or named facility LOC span.
- Products, aircraft models, applications, dates, quantities, and events without
  an organization sense are outside this three-label ontology and receive O.
  Named storms and their transliterations are O, not ORG or LOC.
- Label each occurrence in context. Repeated spellings need not have the same
  type. Article footers may contain author PER and publisher ORG mentions.
- Preserve punctuation belonging to a named mention, including periods in
  dotted abbreviations and matching brackets around an acronym within a full
  institutional mention. Exclude surrounding sentence punctuation. Review
  ambiguous name/alias boundaries explicitly; do not strip punctuation from
  reference labels merely to match model output.

## Provenance and evaluation

New annotations produced in this task are assistant-reviewed drafts. They have
not been independently verified by a Khmer annotator. Preserve that distinction
in manifests and request a native-speaker review before publication-quality claims.
Existing human-reviewed flags in older files are inherited metadata, not new proof.
On 2026-10-21 the project owner, a Khmer reader, reviewed a sample of five labelled news
articles (121 labels) against this policy and found no errors.

Partition complete source articles before fitting, preserve historical test
assignments, and exclude duplicate text across splits. Never use incomplete labels
as if all unmarked tokens were known negatives. Synthetic examples are supplements,
not evidence of real-world accuracy. The earlier repeatedly inspected tests are
development/regression sets. New test articles from the raw collection were exposed
to prior masked-language adaptation; they are held out from supervised labels only.

Select a checkpoint on validation only, then evaluate the locked test once. Include
raw-text character-span evaluation: scoring with gold-split words alone can conceal
segmentation failures at inference. Report misses and false positives, not only F1.
