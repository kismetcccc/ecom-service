"""Executable acceptance checks for the fulfillment-evidence SOP.

Only current-turn, call-bound tool records are inputs. No network, writes,
free-text status inference, or model-supplied evidence is accepted.
"""
from app.schemas.response import SkillAudit, SkillCheck


SKILL_NAME = "fulfillment-evidence"
RULE_VERSION = "1"


def audit_fulfillment(messages, skill_version):
    from app.multi_agent.quality import collect_evidence
    report = SkillAudit(skill=SKILL_NAME, version=skill_version, rule_version=RULE_VERSION)

    def check(code, status, text, refs=()):
        report.checks.append(SkillCheck(code=code, status=status, message=text, evidence_ids=list(refs)))

    try:
        evidence = collect_evidence(messages)
    except (ValueError, TypeError):
        check("INVALID_EVIDENCE", "missing", "工具证据不完整，不能完成交叉核验。")
        return report
    orders = [e for e in evidence if e.tool == "query_order"]
    logistics = [e for e in evidence if e.tool == "query_logistics"]
    if len(orders) != 1 or len(logistics) != 1:
        check("PAIR_REQUIRED", "missing", "需要本轮同一订单的一组订单与物流查询；重复或缺失记录不自动选取。")
        return report
    order, shipment = orders[0], logistics[0]
    refs = [order.evidence_id, shipment.evidence_id]
    o, s = order.data.get("order"), shipment.data.get("logistics")
    if (order.data.get("success") is not True or shipment.data.get("success") is not True
            or not isinstance(o, dict) or not isinstance(s, dict)):
        check("QUERY_FAILED", "missing", "订单或物流查询未成功，不能补猜业务状态。", refs)
        return report
    oid = order.arguments.get("order_id")
    tracking = o.get("tracking_number")
    if (not isinstance(oid, str) or not oid or o.get("order_id") != oid
            or shipment.arguments.get("order_id") != oid or not isinstance(tracking, str)
            or not tracking or tracking != s.get("tracking_number")):
        check("IDENTITY_MISMATCH", "missing", "订单号或运单号无法匹配，不能将两份记录合并。", refs)
        return report
    check("IDENTITY_MATCHED", "verified", "本轮订单号与运单号匹配。", refs)
    os, ls = o.get("status"), s.get("status")
    known_order = {"pending", "purchased", "shipped", "delivered", "refund_processing", "refunded", "cancelled"}
    known_logistics = {"pending", "in_transit", "delivered"}
    if not isinstance(os, str) or not isinstance(ls, str) or os not in known_order or ls not in known_logistics:
        check("UNKNOWN_STATUS", "missing", "存在不支持的状态，不能自动判断状态关系。", refs)
        return report
    elif os == "refund_processing" and ls == "in_transit":
        check("REFUND_IN_TRANSIT", "attention", "退款处理中与运输中可以同时存在，不代表退款到账或物流已拦截。", refs)
    elif (os == "delivered") != (ls == "delivered") or (os in {"pending", "purchased"} and ls == "in_transit"):
        check("STATUS_DIVERGENCE", "attention", "订单与物流阶段存在差异，可能有同步延迟，需核实，不能直接判定丢件。", refs)
    else:
        check("NO_KNOWN_DIVERGENCE", "verified", "未命中当前规则覆盖的状态差异；不代表物流一定正常。", refs)
    if any(isinstance(e.data.get("source"), str) and e.data["source"] in {"demo", "demo_seed"} for e in (order, shipment)):
        check("DEMO_RECORDS", "attention", "记录包含演示数据，不能据历史日期认定当前延误。", refs)
    check("LIVE_STATUS_UNCONFIRMED", "missing", "现有快照不能证明承运商实时状态或停滞原因，需核实最新轨迹。", refs)
    report.status = "review_needed"
    return report
