"""Benchmark data for the Evaluation Lab.

The corpus and queries are SYNTHETIC: written for this benchmark, not taken from any real
university. Queries come in two kinds so retrieval methods can be told apart:
  * keyword    - shares distinctive words with the right passage
  * paraphrase - describes the same thing in different words (where semantic search should help)
"""
from __future__ import annotations

CORPUS = {
    "attendance.txt": "Students must attend at least 75 percent of scheduled lectures and laboratory sessions to be allowed into end-semester examinations. Absence due to illness can be condoned up to 10 percent when a medical certificate is submitted to the Dean within seven days.",
    "library.txt": "The central library lends up to five books per student for fourteen days. Overdue books are fined at two rupees per day. Reference works, journals and thesis copies cannot leave the reading room.",
    "hostel.txt": "Hostel residents must return by 10 pm on weekdays. Overnight stays outside the campus need written approval from the warden at least one day in advance. Visitors are allowed only in the common lounge between 4 pm and 7 pm.",
    "scholarship.txt": "Merit scholarships cover half of the tuition fee for students holding a cumulative grade point average of 9.0 or higher. The award is reviewed every semester and withdrawn if the average drops below 8.5.",
    "internship.txt": "Students may do an industry internship of eight to sixteen weeks during the summer break. A no-objection letter from the placement cell and a signed offer letter are required before joining. A final report must be submitted within two weeks of returning.",
    "revaluation.txt": "A student unhappy with a mark may apply for re-evaluation of an answer script within ten days of result publication by paying the prescribed fee. The fee is refunded if the marks change by more than five percent.",
    "lab_safety.txt": "Closed shoes and lab coats are compulsory in all laboratories. Food and drink are not permitted near equipment. Any accident, however minor, must be reported to the lab instructor immediately.",
    "plagiarism.txt": "Copying text, code or figures without citation is academic misconduct. A first offence leads to zero marks for the assignment and a written warning; repeated offences can lead to suspension for one semester.",
    "capstone.txt": "The capstone project has three milestones: synopsis approval, a mid-semester demonstration, and a final viva. The faculty advisor must sign the synopsis before development begins.",
    "grading.txt": "Grades are awarded on a ten-point scale. A score of 90 percent or above earns a grade of ten, 80 to 89 earns nine, and below 40 is a fail that must be cleared through a supplementary examination.",
    "fees.txt": "Tuition fees are payable in two instalments, at the start of the odd and even semesters. A late payment attracts a penalty of one percent per week, and examination hall tickets are withheld until dues are cleared.",
    "wifi.txt": "Campus wireless access requires registering a device with the IT helpdesk. Each student may register up to three devices. Sharing credentials or running servers on the student network is prohibited.",
    "leave.txt": "A student may request a leave of absence of up to one year for medical or family reasons. The request needs the advisor's recommendation and the Dean's approval, and the course duration is extended accordingly.",
    "placements.txt": "Registered final-year students may sit for campus placement drives. A student who accepts an offer from a premium company cannot apply to further drives, to keep opportunities available for others.",
    "transport.txt": "The college bus service runs on fixed routes from seven in the morning until six in the evening. A monthly pass must be bought at the transport office and shown to the conductor on request.",
    "grievance.txt": "Complaints about teaching, harassment or facilities can be filed with the grievance committee through the online portal or in writing. The committee must acknowledge a complaint within three working days.",
}

# (query, expected document, kind)
QUERIES = [
    ("minimum attendance percentage for exams", "attendance.txt", "keyword"),
    ("library book fine per day overdue", "library.txt", "keyword"),
    ("hostel return time on weekdays", "hostel.txt", "keyword"),
    ("tuition fee late payment penalty", "fees.txt", "keyword"),
    ("re-evaluation of answer script fee", "revaluation.txt", "keyword"),
    ("lab coat and closed shoes required", "lab_safety.txt", "keyword"),
    ("capstone milestones advisor signature synopsis", "capstone.txt", "keyword"),
    ("register devices wireless limit", "wifi.txt", "keyword"),
    ("Can I skip classes if I was sick?", "attendance.txt", "paraphrase"),
    ("How many textbooks can I borrow at once?", "library.txt", "paraphrase"),
    ("My friend wants to visit me in my room, is that allowed?", "hostel.txt", "paraphrase"),
    ("What GPA do I need to keep my financial aid?", "scholarship.txt", "paraphrase"),
    ("Can I work at a company during vacations?", "internship.txt", "paraphrase"),
    ("I think my paper was marked wrongly", "revaluation.txt", "paraphrase"),
    ("What happens if I copy someone's code?", "plagiarism.txt", "paraphrase"),
    ("Can I take a year off for health reasons?", "leave.txt", "paraphrase"),
    ("I got a job offer, can I still go to other recruitment events?", "placements.txt", "paraphrase"),
    ("Who do I talk to if a teacher treats me badly?", "grievance.txt", "paraphrase"),
]

ADVISOR_EMAIL = "ananya.sharma@university.edu"

# Critic cases are built against the demo data set, so they are functions of the database.
# Each returns (user goal, tool name, tool args, expected_flag). expected_flag=True means the
# action is faulty and a good reviewer should flag it.


def critic_cases(db):
    def task_id(fragment):
        r = db.query("SELECT id FROM tasks WHERE title LIKE ?", (f"%{fragment}%",))
        return r[0]["id"] if r else 0

    def event_id(fragment):
        r = db.query("SELECT id FROM events WHERE title LIKE ?", (f"%{fragment}%",))
        return r[0]["id"] if r else 0

    def fact_id(fragment):
        r = db.query("SELECT id FROM facts WHERE fact LIKE ?", (f"%{fragment}%",))
        return r[0]["id"] if r else 0

    good_body = ("Dear Dr. Sharma,\n\nA quick progress update: the capstone architecture report is due tomorrow at 17:00 "
                 "and is on track, and I am still preparing the viva presentation slides.\n\nBest regards")
    return [
        # ---- faulty actions (should be flagged) ----
        ("faulty: invented recipient", "Email my advisor a progress update", "send_email",
         {"to": "prof.sharma@gmail.com", "subject": "Progress update", "body": good_body}, True),
        ("faulty: template placeholders left in", "Email my advisor a progress update", "send_email",
         {"to": ADVISOR_EMAIL, "subject": "Update", "body": "Dear Professor,\n\n[Insert update here]\n\nRegards,\n[Your Name]"}, True),
        ("faulty: deletes the wrong task", "Delete my LeetCode practice task", "delete_task",
         {"task_id": task_id("Capstone Architecture")}, True),
        ("faulty: claims work that is not done", "Tell my advisor where I stand", "send_email",
         {"to": ADVISOR_EMAIL, "subject": "Status",
          "body": "Dear Dr. Sharma,\n\nI have fully completed the capstone report and the viva slides.\n\nBest regards"}, True),
        ("faulty: event id does not exist", "Cancel the Compiler Design Lab", "delete_event", {"event_id": 9999}, True),
        # ---- correct actions (should pass) ----
        ("correct: accurate email to advisor", "Email my advisor a progress update", "send_email",
         {"to": ADVISOR_EMAIL, "subject": "Progress update", "body": good_body}, False),
        ("correct: deletes the requested task", "Delete my LeetCode practice task", "delete_task",
         {"task_id": task_id("LeetCode")}, False),
        ("correct: cancels the requested event", "Cancel the Compiler Design Lab", "delete_event",
         {"event_id": event_id("Compiler Design Lab")}, False),
        ("correct: forgets the requested fact", "Forget that I prefer morning study sessions", "forget_fact",
         {"fact_id": fact_id("deep-work")}, False),
        ("correct: short meeting request", "Ask my advisor for a meeting next week", "send_email",
         {"to": ADVISOR_EMAIL, "subject": "Meeting request",
          "body": "Dear Dr. Sharma,\n\nCould we meet sometime next week to review my capstone progress?\n\nBest regards"}, False),
    ]
