import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from .live_graph import build_live_conversation_title, extract_live_topics
from .models import Event, Pattern, PatternRun, Workspace, WorkspaceMembership, ensure_personal_workspace
from .management.commands.import_user_signals import _extract_signal_topics

User = get_user_model()


class AuthApiTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="jwt-user",
            email="jwt@example.com",
            password="testpass123",
        )
        self.workspace = ensure_personal_workspace(self.user)

    def _seed_workspace_data(self, user=None, workspace=None):
        user = user or self.user
        workspace = workspace or self.workspace
        Event.objects.create(
            workspace=workspace,
            created_by=user,
            occurred_at=timezone.now(),
            source="manual",
            text="Stored event to be deleted.",
            context_tags=["work"],
            people=["manager"],
            emotion="anger",
            intensity=4,
            reaction="Withdrew",
            outcome="Lingering tension",
        )
        run = PatternRun.objects.create(
            workspace=workspace,
            created_by=user,
            params={"source": "test"},
            event_count=1,
        )
        Pattern.objects.create(
            run=run,
            key="test-pattern",
            name="Test Pattern",
            hypothesis="A repeatable test pattern.",
            score=0.9,
            evidence=[{"event_id": "evt-1"}],
        )
        return run

    def test_login_returns_tokens_and_profile(self):
        response = self.client.post(
            reverse("auth-login"),
            {"username": "jwt-user", "password": "testpass123"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["authenticated"])
        self.assertEqual(response.data["user"]["id"], self.user.id)
        self.assertEqual(response.data["workspace"]["id"], str(self.workspace.id))
        self.assertEqual(len(response.data["workspaces"]), 1)
        self.assertEqual(response.data["workspaces"][0]["id"], str(self.workspace.id))
        self.assertIn("accessToken", response.data)
        self.assertIn("refreshToken", response.data)

    def test_me_accepts_bearer_token(self):
        login_response = self.client.post(
            reverse("auth-login"),
            {"username": "jwt-user", "password": "testpass123"},
            format="json",
        )
        access_token = login_response.data["accessToken"]

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")
        response = self.client.get(reverse("auth-session"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["authenticated"])
        self.assertEqual(response.data["user"]["username"], "jwt-user")
        self.assertEqual(response.data["workspace"]["id"], str(self.workspace.id))

    def test_me_can_return_selected_member_workspace(self):
        shared_workspace = Workspace.objects.create(
            name="Shared workspace",
            owner=self.user,
            is_personal=False,
        )
        WorkspaceMembership.objects.create(
            workspace=shared_workspace,
            user=self.user,
            role=Workspace.ROLE_OWNER,
        )
        login_response = self.client.post(
            reverse("auth-login"),
            {"username": "jwt-user", "password": "testpass123"},
            format="json",
        )

        self.client.credentials(
            HTTP_AUTHORIZATION=f"Bearer {login_response.data['accessToken']}",
            HTTP_X_WORKSPACE_ID=str(shared_workspace.id),
        )
        response = self.client.get(reverse("auth-session"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["workspace"]["id"], str(shared_workspace.id))
        self.assertEqual(len(response.data["workspaces"]), 2)

    def test_refresh_issues_new_access_token(self):
        login_response = self.client.post(
            reverse("auth-login"),
            {"username": "jwt-user", "password": "testpass123"},
            format="json",
        )
        original_refresh = login_response.data["refreshToken"]
        response = self.client.post(
            reverse("auth-refresh"),
            {"refresh": original_refresh},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("access", response.data)
        self.assertIn("refresh", response.data)
        self.assertNotEqual(response.data["refresh"], original_refresh)

    def test_logout_requires_authenticated_user(self):
        response = self.client.post(
            reverse("auth-logout"),
            {"refresh": "not-a-real-token"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_logout_blacklists_refresh_token(self):
        login_response = self.client.post(
            reverse("auth-login"),
            {"username": "jwt-user", "password": "testpass123"},
            format="json",
        )
        access_token = login_response.data["accessToken"]
        refresh_token = login_response.data["refreshToken"]

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {access_token}")
        logout_response = self.client.post(
            reverse("auth-logout"),
            {"refresh": refresh_token},
            format="json",
        )

        self.assertEqual(logout_response.status_code, status.HTTP_200_OK)

        refresh_response = self.client.post(
            reverse("auth-refresh"),
            {"refresh": refresh_token},
            format="json",
        )
        self.assertEqual(refresh_response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_logout_rejects_refresh_token_for_different_user(self):
        current_user_login = self.client.post(
            reverse("auth-login"),
            {"username": "jwt-user", "password": "testpass123"},
            format="json",
        )
        other_user = User.objects.create_user(
            username="other-jwt-user",
            password="testpass123",
        )
        ensure_personal_workspace(other_user)
        other_login = self.client.post(
            reverse("auth-login"),
            {"username": "other-jwt-user", "password": "testpass123"},
            format="json",
        )

        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {other_login.data['accessToken']}")
        response = self.client.post(
            reverse("auth-logout"),
            {"refresh": current_user_login.data["refreshToken"]},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_delete_data_requires_auth(self):
        self.client.force_authenticate(user=None)
        response = self.client.delete(reverse("auth-delete-data"))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    @patch("events_api.auth_views.delete_workspace_graph_data", return_value={"deleted_nodes": 12})
    def test_delete_data_clears_workspace_rows_and_graph(self, mock_delete_graph):
        self._seed_workspace_data()
        other_user = User.objects.create_user(username="other-user", password="testpass123")
        other_workspace = ensure_personal_workspace(other_user)
        self._seed_workspace_data(user=other_user, workspace=other_workspace)

        self.client.force_authenticate(user=self.user)
        response = self.client.delete(reverse("auth-delete-data"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["deleted"])
        self.assertEqual(response.data["deleted_events"], 1)
        self.assertEqual(response.data["deleted_pattern_runs"], 1)
        self.assertEqual(response.data["deleted_patterns"], 1)
        self.assertEqual(response.data["deleted_graph_nodes"], 12)
        mock_delete_graph.assert_called_once_with(workspace_id=str(self.workspace.id))

        self.assertEqual(Event.objects.filter(workspace=self.workspace).count(), 0)
        self.assertEqual(PatternRun.objects.filter(workspace=self.workspace).count(), 0)
        self.assertEqual(Pattern.objects.filter(run__workspace=self.workspace).count(), 0)
        self.assertEqual(Event.objects.filter(workspace=other_workspace).count(), 1)

    @patch("events_api.auth_views.delete_workspace_graph_data", side_effect=RuntimeError("neo delete failed"))
    def test_delete_data_rolls_back_postgres_when_graph_delete_fails(self, _mock_delete_graph):
        self._seed_workspace_data()

        self.client.force_authenticate(user=self.user)
        response = self.client.delete(reverse("auth-delete-data"))

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        self.assertEqual(Event.objects.filter(workspace=self.workspace).count(), 1)
        self.assertEqual(PatternRun.objects.filter(workspace=self.workspace).count(), 1)
        self.assertEqual(Pattern.objects.filter(run__workspace=self.workspace).count(), 1)

    @patch("events_api.auth_views.delete_user_graph_data", return_value={"deleted_nodes": 33})
    def test_delete_account_removes_user_account_and_all_data(self, mock_delete_graph):
        self._seed_workspace_data()
        self.client.force_authenticate(user=self.user)

        response = self.client.delete(reverse("auth-delete-account"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["deleted"])
        self.assertEqual(response.data["deleted_username"], "jwt-user")
        self.assertEqual(response.data["deleted_events"], 1)
        self.assertEqual(response.data["deleted_pattern_runs"], 1)
        self.assertEqual(response.data["deleted_patterns"], 1)
        self.assertEqual(response.data["deleted_graph_nodes"], 33)
        mock_delete_graph.assert_called_once_with(
            workspace_ids=[str(self.workspace.id)],
            owner_user_id=self.user.id,
        )

        self.assertFalse(User.objects.filter(username="jwt-user").exists())
        self.assertFalse(Event.objects.filter(workspace=self.workspace).exists())
        self.assertFalse(PatternRun.objects.filter(workspace=self.workspace).exists())
        self.assertFalse(Pattern.objects.exists())

    @patch("events_api.auth_views.delete_user_graph_data", side_effect=RuntimeError("neo delete failed"))
    def test_delete_account_rolls_back_user_delete_when_graph_delete_fails(self, _mock_delete_graph):
        self._seed_workspace_data()
        self.client.force_authenticate(user=self.user)

        response = self.client.delete(reverse("auth-delete-account"))

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        self.assertTrue(User.objects.filter(pk=self.user.pk).exists())
        self.assertEqual(Event.objects.filter(workspace=self.workspace).count(), 1)
        self.assertEqual(PatternRun.objects.filter(workspace=self.workspace).count(), 1)

    @patch("events_api.auth_views.cypher_query")
    def test_export_conversations_downloads_json_bundle(self, mock_cypher_query):
        mock_cypher_query.side_effect = [
            (
                [["conv-1", "Live: conversation", 2, 1773077225.865, 1773077225.865]],
                ["conversation_id", "title", "turn_count", "last_active", "create_time"],
            ),
            (
                [["user", "First message", 1773077225.865], ["assistant", "Reply message", 1773077226.865]],
                ["role", "content", "created_at"],
            ),
        ]

        self.client.force_authenticate(user=self.user)
        response = self.client.get(reverse("auth-export-conversations"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response["Content-Type"].startswith("application/json"))
        self.assertIn("attachment; filename=", response["Content-Disposition"])

        payload = json.loads(response.content)
        self.assertEqual(payload["export_type"], "conversations")
        self.assertEqual(payload["conversation_count"], 1)
        self.assertEqual(payload["conversations"][0]["conversation_id"], "conv-1")
        self.assertEqual(payload["conversations"][0]["messages"][1]["content"], "Reply message")

    @patch("events_api.auth_views.cypher_query")
    def test_export_memory_graph_downloads_svg(self, mock_cypher_query):
        mock_cypher_query.side_effect = [
            (
                [
                    ["user-1", "UserProfile", "You"],
                    ["topic-1", "Topic", "Repair attempts"],
                ],
                ["uid", "kind", "label"],
            ),
            (
                [["user-1", "topic-1", "DISCUSSES"]],
                ["source", "target", "rel_type"],
            ),
        ]

        self.client.force_authenticate(user=self.user)
        response = self.client.get(reverse("auth-export-memory-graph"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response["Content-Type"].startswith("image/svg+xml"))
        self.assertIn("attachment; filename=", response["Content-Disposition"])

        body = response.content.decode("utf-8")
        self.assertIn("<svg", body)
        self.assertIn("ThriveSight Memory Graph", body)
        self.assertIn("Repair attempts", body)


class LiveConversationViewTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="live-user", password="testpass123")
        self.workspace = ensure_personal_workspace(self.user)
        self.client.force_authenticate(user=self.user)

    @patch("events_api.views.LiveConversationService")
    def test_live_conversation_delegates_to_service(self, mock_service_cls):
        mock_service = mock_service_cls.return_value
        mock_service.run.return_value = {
            "response": "That sounds exhausting. What hit you hardest in that moment?",
            "conversation_id": "conv-1234",
            "conversation_title": "Live: Work friction",
            "graph_updated": True,
            "graph_summary": {"updated": True},
            "template": "reflection",
            "signals": {"signal_count": 1},
            "context_assembly": {"entities": {"people": ["manager"]}, "persona": "neutral_observer", "context_packet_length": 19},
        }
        response = self.client.post(
            reverse("live-conversation"),
            {
                "message": "My manager dismissed my idea again.",
                "conversation_id": "conv-1234",
                "template": "reflection",
                "history": [
                    {"role": "user", "content": "I am dreading tomorrow."},
                    {"role": "assistant", "content": "What feels most loaded about it?"},
                ],
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data["response"],
            "That sounds exhausting. What hit you hardest in that moment?",
        )
        self.assertEqual(response.data["conversation_id"], "conv-1234")
        self.assertEqual(response.data["signals"]["signal_count"], 1)
        self.assertEqual(response.data["context_assembly"]["entities"], {"people": ["manager"]})
        mock_service_cls.assert_called_once_with(workspace=self.workspace, user=self.user)
        mock_service.run.assert_called_once_with(
            message="My manager dismissed my idea again.",
            conversation_id="conv-1234",
            template_id="reflection",
            history=[
                {"role": "user", "content": "I am dreading tomorrow."},
                {"role": "assistant", "content": "What feels most loaded about it?"},
            ],
            persona_id=None,
        )


class LiveConversationServiceTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="service-user", password="testpass123")
        self.workspace = ensure_personal_workspace(self.user)

    def _service(self):
        from events_api.live_conversation_service import LiveConversationService

        return LiveConversationService(workspace=self.workspace, user=self.user)

    @patch("events_api.live_conversation_service.LiveConversationService.write_pipeline_trace")
    @patch("events_api.live_conversation_service.LiveConversationService.generate_and_link_signals", return_value={"signal_count": 0})
    @patch(
        "events_api.live_conversation_service.LiveConversationService.persist_conversation",
        return_value={
            "graph_updated": True,
            "graph_summary": {"updated": True},
            "conversation_title": "Live: work friction",
        },
    )
    @patch("events_api.live_conversation_service.LiveConversationService.generate_reply", return_value="AI response")
    @patch("events_api.live_conversation_service.LiveConversationService.build_prompt_context", side_effect=RuntimeError("context exploded"))
    def test_service_uses_fallback_prompt_when_prompt_context_fails(
        self,
        _mock_build_prompt_context,
        mock_generate_reply,
        _mock_persist_conversation,
        _mock_generate_signals,
        _mock_write_trace,
    ):
        result = self._service().run(
            message="I am frustrated with work.",
            conversation_id="conv-1",
            template_id=None,
            history=[],
            persona_id=None,
        )

        self.assertEqual(result["response"], "AI response")
        self.assertEqual(result["context_assembly"]["entities"], {})
        self.assertEqual(result["context_assembly"]["context_packet_length"], 0)
        called_prompt = mock_generate_reply.call_args.kwargs["system_prompt"]
        self.assertIn("warm and perceptive AI companion", called_prompt)

    @patch("events_api.live_conversation_service.LiveConversationService.write_pipeline_trace")
    @patch("events_api.live_conversation_service.LiveConversationService.generate_and_link_signals", return_value={"signal_count": 0})
    @patch(
        "events_api.live_conversation_service.LiveConversationService.persist_conversation",
        return_value={
            "graph_updated": True,
            "graph_summary": {"updated": True},
            "conversation_title": "Live: work friction",
        },
    )
    @patch("events_api.live_conversation_service.LiveConversationService.generate_reply", side_effect=RuntimeError("llm unavailable"))
    @patch(
        "events_api.live_conversation_service.LiveConversationService.build_prompt_context",
        return_value={
            "system_prompt": "assembled prompt",
            "entities": {"people": ["manager"]},
            "persona_id": "neutral_observer",
            "context_packet": "packet",
        },
    )
    def test_service_returns_graceful_reply_when_generation_fails(
        self,
        _mock_build_prompt_context,
        _mock_generate_reply,
        _mock_persist_conversation,
        _mock_generate_signals,
        _mock_write_trace,
    ):
        result = self._service().run(
            message="I am frustrated with work.",
            conversation_id="conv-2",
            template_id=None,
            history=[],
            persona_id=None,
        )

        self.assertIn("having trouble connecting", result["response"])

    @patch("events_api.live_conversation_service.LiveConversationService.write_pipeline_trace")
    @patch("events_api.live_conversation_service.LiveConversationService.generate_and_link_signals", return_value={"signal_count": 0})
    @patch("events_api.live_conversation_service.LiveConversationService.persist_conversation", side_effect=RuntimeError("neo failed"))
    @patch("events_api.live_conversation_service.LiveConversationService.generate_reply", return_value="AI response")
    @patch(
        "events_api.live_conversation_service.LiveConversationService.build_prompt_context",
        return_value={
            "system_prompt": "assembled prompt",
            "entities": {"people": ["manager"]},
            "persona_id": "neutral_observer",
            "context_packet": "packet",
        },
    )
    def test_service_returns_response_when_graph_write_fails(
        self,
        _mock_build_prompt_context,
        _mock_generate_reply,
        _mock_persist_conversation,
        _mock_generate_signals,
        _mock_write_trace,
    ):
        result = self._service().run(
            message="I am frustrated with work.",
            conversation_id="conv-3",
            template_id=None,
            history=[],
            persona_id=None,
        )

        self.assertEqual(result["response"], "AI response")
        self.assertFalse(result["graph_updated"])
        self.assertIsNone(result["graph_summary"])

    @patch("events_api.live_conversation_service.LiveConversationService.write_pipeline_trace")
    @patch("events_api.live_conversation_service.LiveConversationService.generate_and_link_signals", side_effect=RuntimeError("signals failed"))
    @patch(
        "events_api.live_conversation_service.LiveConversationService.persist_conversation",
        return_value={
            "graph_updated": True,
            "graph_summary": {"updated": True},
            "conversation_title": "Live: work friction",
        },
    )
    @patch("events_api.live_conversation_service.LiveConversationService.generate_reply", return_value="AI response")
    @patch(
        "events_api.live_conversation_service.LiveConversationService.build_prompt_context",
        return_value={
            "system_prompt": "assembled prompt",
            "entities": {"people": ["manager"]},
            "persona_id": "neutral_observer",
            "context_packet": "packet",
        },
    )
    def test_service_returns_response_when_signal_generation_fails(
        self,
        _mock_build_prompt_context,
        _mock_generate_reply,
        _mock_persist_conversation,
        _mock_generate_signals,
        _mock_write_trace,
    ):
        result = self._service().run(
            message="I am frustrated with work.",
            conversation_id="conv-4",
            template_id=None,
            history=[],
            persona_id=None,
        )

        self.assertEqual(result["response"], "AI response")
        self.assertEqual(result["signals"]["signal_count"], 0)
        self.assertEqual(result["signals"]["addresses"], [])
        self.assertEqual(result["signals"]["emotions"], [])
        self.assertEqual(result["signals"]["observation_biases"], [])
        self.assertEqual(result["signals"]["error"], "signals failed")


class GraphQueryNormalizationTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="graph-user", password="testpass123")
        self.workspace = ensure_personal_workspace(self.user)
        self.client.force_authenticate(user=self.user)

    @patch("events_api.views.cypher_query")
    def test_conversation_list_returns_normalized_conversations_key(self, mock_cypher_query):
        mock_cypher_query.return_value = (
            [["conv-1", "Work friction", 6, "2026-03-09T10:00:00Z", "2026-03-08T09:00:00Z"]],
            ["conversation_id", "title", "turn_count", "last_active", "create_time"],
        )

        response = self.client.post(
            reverse("graph-query"),
            {"query_type": "conversation_list"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("conversations", response.data)
        self.assertEqual(response.data["conversations"][0]["conversation_id"], "conv-1")
        self.assertEqual(response.data["results"][0]["conversation_id"], "conv-1")

    @patch("events_api.views.cypher_query")
    def test_conversation_history_returns_normalized_turns_key(self, mock_cypher_query):
        mock_cypher_query.return_value = (
            [["user", "I am overwhelmed.", "2026-03-09T10:00:00Z"]],
            ["role", "content", "created_at"],
        )

        response = self.client.post(
            reverse("graph-query"),
            {"query_type": "conversation_history", "conversation_id": "conv-1"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("turns", response.data)
        self.assertEqual(response.data["turns"][0]["role"], "user")
        self.assertEqual(response.data["results"][0]["content"], "I am overwhelmed.")

    @patch("events_api.views.cypher_query")
    def test_emotional_graph_returns_structured_node_and_edge_fields(self, mock_cypher_query):
        mock_cypher_query.side_effect = [
            ([[2]], ["cnt"]),
            ([[0]], ["cnt"]),
            ([[0]], ["cnt"]),
            ([[0]], ["cnt"]),
            ([[0]], ["cnt"]),
            (
                [
                    ["emo-1", {"name": "Anxiety", "valence": 2}, 3, ["SA(work, manager, dismissal, monday)|anxiety"]],
                    ["emo-2", {"name": "Relief", "valence": 4}, 1, ["SA(home, self, recovery, night)|relief"]],
                ],
                ["uid", "node_properties", "signal_count", "signal_addresses"],
            ),
            (
                [["emo-1", "emo-2", 2]],
                ["source", "target", "weight"],
            ),
        ]

        response = self.client.post(
            reverse("graph-query"),
            {"query_type": "emotional_graph", "categories": ["Emotion"]},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["nodes"][0]["name"], "Anxiety")
        self.assertEqual(response.data["nodes"][0]["weight"], 3)
        self.assertEqual(response.data["edges"][0]["relationship"], "CONNECTED_THROUGH_SIGNALS")


class ApiSpecTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="alice", password="testpass123")
        self.workspace = ensure_personal_workspace(self.user)
        self.client.force_authenticate(user=self.user)

    def _event_payload(self, **overrides):
        payload = {
            "occurred_at": timezone.now().isoformat(),
            "source": "manual",
            "text": "Difficult meeting with manager about roadmap delays.",
            "context_tags": ["work"],
            "people": ["manager"],
            "emotion": "anger",
            "intensity": 4,
            "reaction": "Raised voice",
            "outcome": "Tense end to meeting",
        }
        payload.update(overrides)
        return payload

    def test_auth_is_required(self):
        self.client.force_authenticate(user=None)
        response = self.client.get(reverse("events-list-create-delete"))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_events_can_be_scoped_to_selected_member_workspace(self):
        shared_owner = User.objects.create_user(username="shared-owner", password="testpass123")
        shared_workspace = Workspace.objects.create(
            name="Shared workspace",
            owner=shared_owner,
            is_personal=False,
        )
        WorkspaceMembership.objects.create(
            workspace=shared_workspace,
            user=shared_owner,
            role=Workspace.ROLE_OWNER,
        )
        WorkspaceMembership.objects.create(
            workspace=shared_workspace,
            user=self.user,
            role=Workspace.ROLE_MEMBER,
        )
        Event.objects.create(
            workspace=shared_workspace,
            created_by=shared_owner,
            occurred_at=timezone.now(),
            source="manual",
            text="Shared workspace event",
        )
        Event.objects.create(
            workspace=self.workspace,
            created_by=self.user,
            occurred_at=timezone.now(),
            source="manual",
            text="Personal workspace event",
        )

        response = self.client.get(
            reverse("events-list-create-delete"),
            HTTP_X_WORKSPACE_ID=str(shared_workspace.id),
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual([item["text"] for item in response.data], ["Shared workspace event"])

    def test_event_post_accepts_workspace_selection_from_request_body(self):
        shared_owner = User.objects.create_user(username="body-owner", password="testpass123")
        shared_workspace = Workspace.objects.create(
            name="Body-selected workspace",
            owner=shared_owner,
            is_personal=False,
        )
        WorkspaceMembership.objects.create(
            workspace=shared_workspace,
            user=shared_owner,
            role=Workspace.ROLE_OWNER,
        )
        WorkspaceMembership.objects.create(
            workspace=shared_workspace,
            user=self.user,
            role=Workspace.ROLE_MEMBER,
        )

        response = self.client.post(
            reverse("events-list-create-delete"),
            {
                **self._event_payload(text="Stored in selected shared workspace"),
                "workspace_id": str(shared_workspace.id),
            },
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        event = Event.objects.get(id=response.data["id"])
        self.assertEqual(event.workspace_id, shared_workspace.id)

    def test_workspace_selection_rejects_inaccessible_workspace(self):
        other_user = User.objects.create_user(username="forbidden-owner", password="testpass123")
        inaccessible_workspace = Workspace.objects.create(
            name="Forbidden workspace",
            owner=other_user,
            is_personal=False,
        )
        WorkspaceMembership.objects.create(
            workspace=inaccessible_workspace,
            user=other_user,
            role=Workspace.ROLE_OWNER,
        )

        response = self.client.get(
            reverse("events-list-create-delete"),
            HTTP_X_WORKSPACE_ID=str(inaccessible_workspace.id),
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_create_and_filter_events(self):
        response = self.client.post(reverse("events-list-create-delete"), self._event_payload(), format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(emotion="sadness", context_tags=["family"], text="Argument at home"),
            format="json",
        )

        emotion_filtered = self.client.get(reverse("events-list-create-delete"), {"emotion": "anger"})
        self.assertEqual(emotion_filtered.status_code, status.HTTP_200_OK)
        self.assertEqual(len(emotion_filtered.data), 1)

        tag_filtered = self.client.get(reverse("events-list-create-delete"), {"tag": "work"})
        self.assertEqual(tag_filtered.status_code, status.HTTP_200_OK)
        self.assertEqual(len(tag_filtered.data), 1)

    def test_recompute_patterns_and_latest_patterns(self):
        self.client.post(reverse("events-list-create-delete"), self._event_payload(text="Work conflict one"), format="json")
        self.client.post(reverse("events-list-create-delete"), self._event_payload(text="Work conflict two"), format="json")
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(context_tags=["family"], emotion="sadness", text="Family stressor"),
            format="json",
        )

        recompute = self.client.post(reverse("patterns-recompute"), {"max_patterns": 7, "evidence_per_pattern": 5}, format="json")
        self.assertEqual(recompute.status_code, status.HTTP_200_OK)
        self.assertIn("run_id", recompute.data)
        self.assertGreaterEqual(len(recompute.data["patterns"]), 1)

        latest = self.client.get(reverse("patterns-list"))
        self.assertEqual(latest.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(latest.data), 1)

        first_pattern_id = latest.data[0]["id"]
        detail = self.client.get(reverse("patterns-detail", kwargs={"pk": first_pattern_id}))
        self.assertEqual(detail.status_code, status.HTTP_200_OK)
        self.assertIn("evidence", detail.data)

    def test_recompute_is_deterministic_for_same_input(self):
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(text="Conflict with manager after standup", context_tags=["work"], emotion="anger"),
            format="json",
        )
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(text="Stress after deadline slip", context_tags=["work"], emotion="anger"),
            format="json",
        )
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(text="Family check-in felt tense", context_tags=["family"], emotion="sadness"),
            format="json",
        )

        params = {"max_patterns": 7, "evidence_per_pattern": 5}
        first = self.client.post(reverse("patterns-recompute"), params, format="json")
        second = self.client.post(reverse("patterns-recompute"), params, format="json")

        self.assertEqual(first.status_code, status.HTTP_200_OK)
        self.assertEqual(second.status_code, status.HTTP_200_OK)

        first_signatures = [
            (item["key"], item["name"], item["score"], [ev["event_id"] for ev in item["evidence"]])
            for item in first.data["patterns"]
        ]
        second_signatures = [
            (item["key"], item["name"], item["score"], [ev["event_id"] for ev in item["evidence"]])
            for item in second.data["patterns"]
        ]
        self.assertEqual(first_signatures, second_signatures)

    def test_recompute_handles_missing_optional_event_fields(self):
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(
                context_tags=[],
                emotion=None,
                intensity=None,
                reaction=None,
                outcome=None,
                people=[],
                text="No optional metadata on this event.",
            ),
            format="json",
        )
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(
                context_tags=["work"],
                emotion="",
                intensity=None,
                text="Another sparse event in work context.",
            ),
            format="json",
        )

        response = self.client.post(
            reverse("patterns-recompute"),
            {"max_patterns": 5, "evidence_per_pattern": 3},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data["patterns"]), 1)
        for pattern in response.data["patterns"]:
            self.assertIn("name", pattern)
            self.assertIn("hypothesis", pattern)
            self.assertIn("score", pattern)
            self.assertIn("evidence", pattern)
            self.assertLessEqual(len(pattern["evidence"]), 3)

    def test_delete_events_also_clears_derived_patterns(self):
        self.client.post(reverse("events-list-create-delete"), self._event_payload(), format="json")
        self.client.post(reverse("patterns-recompute"), {}, format="json")

        self.assertGreater(Event.objects.count(), 0)
        self.assertGreater(PatternRun.objects.count(), 0)
        self.assertGreater(Pattern.objects.count(), 0)

        delete_response = self.client.delete(reverse("events-list-create-delete"))
        self.assertEqual(delete_response.status_code, status.HTTP_200_OK)
        self.assertEqual(delete_response.data["deleted_events"], 1)
        self.assertEqual(Event.objects.filter(workspace=self.workspace).count(), 0)
        self.assertEqual(PatternRun.objects.filter(workspace=self.workspace).count(), 0)
        self.assertEqual(Pattern.objects.count(), 0)

    def test_ask_returns_insufficient_evidence_when_fewer_than_two_citations(self):
        self.client.post(reverse("events-list-create-delete"), self._event_payload(text="One isolated event"), format="json")

        response = self.client.post(reverse("ask"), {"question": "Why am I upset at work?"}, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("Insufficient", response.data["answer"])
        self.assertLess(len(response.data["citations"]), 2)

    def test_ask_returns_citations_when_enough_relevant_events_exist(self):
        e1 = self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(text="Meeting conflict with manager at work", emotion="anger"),
            format="json",
        )
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(text="Another meeting conflict created tension", emotion="anger"),
            format="json",
        )

        response = self.client.post(
            reverse("ask"),
            {
                "question": "Why do meeting conflicts keep happening?",
                "focus_event_id": e1.data["id"],
            },
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data["citations"]), 2)
        self.assertIn("used_events", response.data)

    def test_ask_uses_reaction_outcome_tag_and_emotion_fields_for_relevance(self):
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(
                text="Brief check-in.",
                context_tags=["work"],
                emotion="anger",
                reaction="Withdrew in meetings",
                outcome="Deadline slipped",
            ),
            format="json",
        )
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(
                text="Another short note.",
                context_tags=["work"],
                emotion="anger",
                reaction="Withdrew in meetings",
                outcome="Deadline slipped",
            ),
            format="json",
        )
        self.client.post(
            reverse("events-list-create-delete"),
            self._event_payload(
                text="Completely unrelated family dinner.",
                context_tags=["family"],
                emotion="joy",
                reaction="Relaxed",
                outcome="Connected",
            ),
            format="json",
        )

        response = self.client.post(
            reverse("ask"),
            {"question": "Why do work anger deadline problems repeat?"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data["citations"]), 2)
        self.assertIn("recurring context tag", response.data["answer"])
        self.assertIn("recurring emotion", response.data["answer"])
        self.assertIn("recurring reaction/outcome", response.data["answer"])

    def test_ask_limits_citations_to_top_k(self):
        for index in range(7):
            self.client.post(
                reverse("events-list-create-delete"),
                self._event_payload(
                    text=f"Meeting friction entry {index}",
                    context_tags=["work"],
                    emotion="anger",
                    reaction="Raised voice",
                    outcome="Meeting ended abruptly",
                ),
                format="json",
            )

        response = self.client.post(
            reverse("ask"),
            {"question": "Why does meeting friction keep happening at work?"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["citations"]), 5)

    def test_events_are_scoped_to_current_workspace(self):
        other_user = User.objects.create_user(username="bob", password="testpass123")
        other_workspace = ensure_personal_workspace(other_user)
        Event.objects.create(
            workspace=other_workspace,
            created_by=other_user,
            occurred_at=timezone.now(),
            source="manual",
            text="Other user's private event",
        )

        self.client.post(reverse("events-list-create-delete"), self._event_payload(), format="json")

        response = self.client.get(reverse("events-list-create-delete"))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertNotIn("Other user's private event", [item["text"] for item in response.data])

    def test_conversation_scoped_graph_query_requires_conversation_id(self):
        response = self.client.post(
            reverse("graph-query"),
            {"query_type": "conversation_topics"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("conversation_id", response.data["detail"])

    @patch("events_api.graph_tests.cypher_query")
    def test_graph_tests_returns_graph_backed_cards_and_applies_context_filter(self, mocked_cypher_query):
        now = timezone.now()
        mocked_cypher_query.return_value = (
            [
                [
                    "sig-work",
                    now.isoformat(),
                    "Work anxiety turned into procrastination and shame.",
                    "SA(work, manager, procrastination, monday)",
                    json.dumps(
                        [
                            {"emotion": "anxiety", "intensity": 7},
                            {"emotion": "shame", "intensity": 6},
                        ]
                    ),
                    7,
                    ["work"],
                    ["manager"],
                    ["anxiety", "shame"],
                    [],
                    ["procrastination"],
                    ["later_regret"],
                    [],
                ],
                [
                    "sig-repair",
                    now.isoformat(),
                    "Relationship repair brought relief and clarity.",
                    "SA(relationship, maya, apology, tuesday)",
                    json.dumps(
                        [
                            {"emotion": "relief", "intensity": 5},
                            {"emotion": "clarity", "intensity": 4},
                        ]
                    ),
                    5,
                    ["relationship", "repair"],
                    ["maya"],
                    ["relief", "clarity"],
                    [],
                    ["apology"],
                    ["reconnection"],
                    [],
                ],
            ],
            [
                "id",
                "occurred_at",
                "text",
                "signal_address",
                "emotions_payload",
                "legacy_intensity",
                "contexts",
                "people",
                "emotion_names",
                "behavior_names",
                "action_names",
                "outcome_names",
                "turn_previews",
            ],
        )

        response = self.client.get(
            reverse("graph-tests"),
            {
                "time_range": "all",
                "contexts": "work",
                "compare_by": "time_of_day",
                "pattern_trend": "all",
            },
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["cards"]), 13)
        self.assertEqual(response.data["filters"]["contexts"], ["work"])
        self.assertEqual(response.data["filters"]["pattern_trend"], "all")
        self.assertEqual(response.data["meta"]["event_count"], 1)
        self.assertTrue(response.data["meta"]["graph_available"])
        self.assertEqual(response.data["cards"][0]["kind"], "matrix")
        self.assertTrue(any(card["kind"] == "chord" for card in response.data["cards"]))
        self.assertEqual(response.data["cards"][-1]["kind"], "timeline-graph")


class LiveGraphHelpersTests(APITestCase):
    def test_extract_live_topics_filters_stop_words_and_orders_by_count(self):
        topics = extract_live_topics(
            "Work stress keeps showing up in work planning and stress spirals."
        )
        topic_counts = {topic["word"]: topic["count"] for topic in topics}

        self.assertGreaterEqual(len(topics), 2)
        self.assertEqual(topic_counts["stress"], 2)
        self.assertEqual(topic_counts["work"], 2)
        self.assertTrue(all(topic["word"] not in {"and", "the", "in"} for topic in topics))

    def test_build_live_conversation_title_trims_and_falls_back(self):
        self.assertEqual(build_live_conversation_title("   "), "Live reflection")
        self.assertTrue(
            build_live_conversation_title("I keep replaying that meeting with my manager.").startswith("Live:")
        )

    def test_extract_signal_topics_blends_text_and_structured_fields(self):
        topics = _extract_signal_topics(
            {
                "id": "sig-1",
                "title": "Money talk made me shut down",
                "text": "Money stress kept spiking and I shut down during the budgeting talk.",
                "contexts": ["home", "money"],
                "actions": ["shutdown"],
                "emotions": ["anxiety", "embarrassment"],
            }
        )
        topic_counts = {topic["word"]: topic["count"] for topic in topics}

        self.assertIn("money", topic_counts)
        self.assertIn("shutdown", topic_counts)
        self.assertGreaterEqual(topic_counts["money"], 3)


class ImportUserSignalsCommandTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="jack", password="testpass123")
        self.workspace = ensure_personal_workspace(self.user)

    def test_command_imports_events_for_user_workspace(self):
        payload = [
            {
                "id": "sig-001",
                "date": "2026-01-03T21:10:00Z",
                "title": "Dinner felt tense",
                "text": "Dinner with Maya felt off and I withdrew.",
                "people": ["Maya"],
                "contexts": ["home", "relationship"],
                "actions": ["avoidance"],
                "emotions": ["anxiety", "guilt"],
                "outcome": "distance after dinner",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "signals.json"
            file_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch("events_api.management.commands.import_user_signals.cypher_query", return_value=([], [])) as mocked:
                call_command("import_user_signals", username="jack", file=str(file_path))

        event = Event.objects.get(workspace=self.workspace)
        self.assertEqual(event.created_by, self.user)
        self.assertEqual(event.source, Event.SOURCE_IMPORT)
        self.assertIn("Dinner felt tense", event.text)
        self.assertIn("import_id:sig-001", event.context_tags)
        self.assertIn("action:avoidance", event.context_tags)
        self.assertEqual(event.people, ["Maya"])
        self.assertEqual(event.emotion, "anxiety")
        self.assertEqual(event.reaction, "avoidance")
        self.assertEqual(event.outcome, "distance after dinner")
        self.assertTrue(mocked.called)

    def test_command_updates_existing_imported_event_instead_of_duplicating(self):
        payload = [
            {
                "id": "sig-001",
                "date": "2026-01-03T21:10:00Z",
                "title": "Original title",
                "text": "Original text",
                "people": ["Maya"],
                "contexts": ["home"],
                "actions": ["avoidance"],
                "emotions": ["anxiety"],
                "outcome": "original outcome",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "signals.json"
            file_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch("events_api.management.commands.import_user_signals.cypher_query", return_value=([], [])):
                call_command("import_user_signals", username="jack", file=str(file_path), skip_graph=True)

                payload[0]["title"] = "Updated title"
                payload[0]["text"] = "Updated text"
                file_path.write_text(json.dumps(payload), encoding="utf-8")
                call_command("import_user_signals", username="jack", file=str(file_path), skip_graph=True)

        self.assertEqual(Event.objects.filter(workspace=self.workspace).count(), 1)
        event = Event.objects.get(workspace=self.workspace)
        self.assertIn("Updated title", event.text)
