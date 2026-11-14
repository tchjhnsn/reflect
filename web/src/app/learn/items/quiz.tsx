"use client";

import { useState, useTransition } from "react";
import { QUIZ, QUIZ_KINDS, gaps, type Checked, type Question } from "@reflect/core/quiz";
import type { Passage } from "@reflect/core/courses";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { checkAnswerAction, quizAction } from "../actions";

type Quiz = { id: string; questions: Question[]; checked: Checked[] };

function One({ quizId, seq, q, checked, passage, onChecked }: { quizId: string; seq: number; q: Question; checked: Checked | undefined; passage: Passage | null; onChecked: (c: Checked) => void }) {
  const [answer, setAnswer] = useState("");
  const [shown, setShown] = useState(false);
  const [pending, run] = useTransition();
  const open = shown || Boolean(checked);
  const keep = (knew: boolean) => run(async () => {
    if (await checkAnswerAction(quizId, seq, answer, knew)) onChecked({ seq, answer: answer.trim(), knew });
  });
  return (
    <li className="space-y-3 rounded-lg border p-4">
      <div className="flex items-start gap-2">
        <span className="text-muted-foreground w-5 shrink-0 text-right text-sm tabular-nums">{seq + 1}</span>
        <div className="min-w-0 flex-1 space-y-3">
          <p className="font-medium leading-snug">{q.question} <Badge variant="outline" className="ml-1 align-middle">{QUIZ_KINDS[q.kind]}</Badge></p>
          {checked ? (
            checked.answer ? <p className="text-sm leading-relaxed"><span className="text-muted-foreground">You wrote: </span>{checked.answer}</p> : null
          ) : (
            <Textarea aria-label={`Your answer to question ${seq + 1}`} value={answer} onChange={(e) => setAnswer(e.target.value)} rows={3} maxLength={QUIZ.answerByPerson} disabled={shown} placeholder="Answer from memory first" />
          )}
          {open ? (
            <div className="bg-muted/40 space-y-2 rounded-md p-3 text-sm leading-relaxed">
              <p><span className="font-medium">The lecture: </span>{q.answer}</p>
              {passage ? <blockquote className="text-muted-foreground border-l-2 pl-3">{passage.text.length > 500 ? `${passage.text.slice(0, 497)}...` : passage.text}</blockquote> : null}
            </div>
          ) : null}
          {checked ? (
            <p className={`text-sm font-medium ${checked.knew ? "" : "text-amber-700 dark:text-amber-400"}`}>{checked.knew ? "You knew it" : "Not yet"}</p>
          ) : shown ? (
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" disabled={pending} onClick={() => keep(true)}>I knew it</Button>
              <Button size="sm" variant="outline" disabled={pending} onClick={() => keep(false)}>Not yet</Button>
            </div>
          ) : (
            <Button size="sm" variant="outline" onClick={() => setShown(true)}>Check</Button>
          )}
        </div>
      </div>
    </li>
  );
}

/**
 * Quiz yourself on a lecture: questions a model wrote from its transcript on what it said and why, each answered from
 * memory, then set beside the lecture's own answer and passage. You say whether you knew it. Only you see it, and
 * nothing is graded; people, not a model, examine you.
 */
export function LectureQuiz({ item, unit, transcript, start }: { item: string; unit: string; transcript: Passage[]; start: Quiz | null }) {
  const [quiz, setQuiz] = useState<Quiz | null>(start);
  const [error, setError] = useState<string | null>(null);
  const [pending, run] = useTransition();
  const make = () => run(async () => {
    setError(null);
    const r = await quizAction(item, unit);
    if ("error" in r) setError(r.error);
    else setQuiz(r);
  });
  const done = quiz ? quiz.checked.length === quiz.questions.length : false;
  const missing = quiz ? gaps(quiz.questions, quiz.checked) : [];
  return (
    <section id="quiz" className="scroll-mt-20 space-y-4">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">Quiz yourself</h2>
        <p className="text-muted-foreground text-sm">Questions on what the lecture said and why, written by a model from its transcript. Answer from memory, then check. Only you see this, and it is not graded.</p>
      </div>
      {quiz ? (
        <>
          <ol className="space-y-3">
            {quiz.questions.map((q, seq) => (
              <One key={`${quiz.id}-${seq}`} quizId={quiz.id} seq={seq} q={q} checked={quiz.checked.find((c) => c.seq === seq)} passage={q.passage !== null ? transcript[q.passage] ?? null : null}
                onChecked={(c) => setQuiz((cur) => (cur ? { ...cur, checked: [...cur.checked, c] } : cur))} />
            ))}
          </ol>
          {done ? (
            <div className="space-y-2 rounded-lg border p-4 text-sm">
              <p className="font-medium">{missing.length ? `What you do not know yet (${missing.length} of ${quiz.questions.length})` : `You knew all ${quiz.questions.length}.`}</p>
              {missing.length ? <ul className="list-disc space-y-1 pl-5">{missing.map((m) => <li key={m.seq}>{m.question}</li>)}</ul> : null}
            </div>
          ) : null}
          <Button size="sm" variant="outline" disabled={pending} onClick={make}>{pending ? "Writing a new quiz..." : "A new quiz"}</Button>
        </>
      ) : (
        <Button size="sm" disabled={pending} onClick={make}>{pending ? "Writing the quiz..." : "Quiz yourself on this lecture"}</Button>
      )}
      {error ? <p className="text-destructive text-sm">{error}</p> : null}
    </section>
  );
}
