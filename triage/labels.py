"""The approved label list. The model may only answer with these; anything else is rejected.

The labels are the ones used by the public Bitext customer-support dataset. In a client project
this file is replaced by the client's own routing scheme and the model is trained again.
"""
from __future__ import annotations

# intent -> category. An intent belongs to exactly one category, which gives a free
# consistency check on the model's output.
INTENT_TO_CATEGORY: dict[str, str] = {
    "create_account": "ACCOUNT",
    "delete_account": "ACCOUNT",
    "edit_account": "ACCOUNT",
    "recover_password": "ACCOUNT",
    "registration_problems": "ACCOUNT",
    "switch_account": "ACCOUNT",
    "check_cancellation_fee": "CANCEL",
    "contact_customer_service": "CONTACT",
    "contact_human_agent": "CONTACT",
    "delivery_options": "DELIVERY",
    "delivery_period": "DELIVERY",
    "complaint": "FEEDBACK",
    "review": "FEEDBACK",
    "check_invoice": "INVOICE",
    "get_invoice": "INVOICE",
    "cancel_order": "ORDER",
    "change_order": "ORDER",
    "place_order": "ORDER",
    "track_order": "ORDER",
    "check_payment_methods": "PAYMENT",
    "payment_issue": "PAYMENT",
    "check_refund_policy": "REFUND",
    "get_refund": "REFUND",
    "track_refund": "REFUND",
    "change_shipping_address": "SHIPPING",
    "set_up_shipping_address": "SHIPPING",
    "newsletter_subscription": "SUBSCRIPTION",
}

INTENTS: list[str] = list(INTENT_TO_CATEGORY)
CATEGORIES: list[str] = sorted(set(INTENT_TO_CATEGORY.values()))
