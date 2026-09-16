import os
import unittest
from email.message import EmailMessage
from types import SimpleNamespace
from unittest.mock import patch

import job_email
from langgraph.graph import END


class EmailWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.dict(
            os.environ,
            {
                "EMAIL_IMAP_HOST": "imap.example.com",
                "EMAIL_SMTP_HOST": "smtp.example.com",
                "EMAIL_USERNAME": "me@example.com",
                "EMAIL_PASSWORD": "test-password",
            },
        )
        self.settings.start()
        self.addCleanup(self.settings.stop)

    @patch("job_email.imaplib.IMAP4_SSL")
    def test_read_email_uses_peek_and_extracts_reply_fields(self, imap_ssl):
        message = EmailMessage()
        message["From"] = "Recruiter <recruiter@example.com>"
        message["Reply-To"] = "replies@example.com"
        message["Subject"] = "Interview invitation"
        message["Message-ID"] = "<abc@example.com>"
        message.set_content("Please interview with us.")
        mailbox = imap_ssl.return_value.__enter__.return_value
        mailbox.select.return_value = ("OK", [b""])
        mailbox.uid.side_effect = [
            ("OK", [b"12 34"]),
            ("OK", [(b"34 (BODY[] ...)", message.as_bytes())]),
        ]

        result = job_email.read_email.invoke({})

        self.assertEqual(result["uid"], "34")
        self.assertEqual(result["reply_to"], "replies@example.com")
        self.assertIn("Please interview", result["body"])
        mailbox.select.assert_called_once_with("INBOX", readonly=True)
        mailbox.uid.assert_any_call("fetch", "34", "(BODY.PEEK[])")

    @patch("job_email.smtplib.SMTP")
    def test_send_email_builds_threaded_message(self, smtp_class):
        smtp = smtp_class.return_value.__enter__.return_value
        result = job_email.send_email.invoke(
            {
                "to_address": "recruiter@example.com",
                "subject": "Re: Interview invitation",
                "body": "Thank you. I am interested.",
                "in_reply_to": "<abc@example.com>",
            }
        )

        self.assertIn("recruiter@example.com", result)
        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with("me@example.com", "test-password")
        sent = smtp.send_message.call_args.args[0]
        self.assertEqual(sent["To"], "recruiter@example.com")
        self.assertEqual(sent["In-Reply-To"], "<abc@example.com>")
        self.assertEqual(sent["References"], "<abc@example.com>")

    def test_classification_and_route(self):
        email = {
            "from": "Recruiter <recruiter@example.com>",
            "reply_to": "recruiter@example.com",
            "subject": "Next steps",
            "body": "Please schedule an interview.",
            "message_id": "<abc@example.com>",
            "references": "",
        }
        with patch("job_email.ChatGroq") as model_class:
            model_class.return_value.invoke.return_value = SimpleNamespace(
                content="moving_forward"
            )
            update = job_email.classify_node({"messages": [], "email": email})

        self.assertEqual(update["classification"], "moving_forward")
        self.assertEqual(job_email.route(update), "email_agent")
        self.assertIn("recruiter@example.com", update["messages"][0]["content"])
        self.assertEqual(job_email.route({"classification": "rejected"}), END)
        self.assertEqual(job_email.route({"classification": "other"}), END)


if __name__ == "__main__":
    unittest.main()
