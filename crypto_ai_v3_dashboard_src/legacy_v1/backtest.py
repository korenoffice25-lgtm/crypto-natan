from dataclasses import dataclass
import pandas as pd
from risk import size_position
from config import SETTINGS

@dataclass
class Position:
    qty: float
    entry: float
    stop: float
    target: float
    entry_time: object

def run_backtest(df: pd.DataFrame, starting_cash: float | None = None):
    cfg = SETTINGS
    cash = starting_cash or cfg.starting_cash
    position = None
    trades = []
    equity_curve = []
    current_day = None
    day_start_equity = cash
    halted_today = False

    for _, row in df.iterrows():
        ts = row["timestamp"]
        price = float(row["close"])
        atr = float(row["atr_14"])
        prob = float(row["prob_up"])

        equity = cash + (position.qty * price if position else 0)
        day = ts.date()

        if current_day != day:
            current_day = day
            day_start_equity = equity
            halted_today = False

        if equity <= day_start_equity * (1 - cfg.max_daily_loss_pct):
            halted_today = True

        if position:
            exit_price = None
            reason = None

            # Conservative assumption when both stop and target are touched:
            # stop is processed first.
            if float(row["low"]) <= position.stop:
                exit_price = position.stop
                reason = "STOP"
            elif float(row["high"]) >= position.target:
                exit_price = position.target
                reason = "TARGET"
            elif prob <= cfg.exit_probability:
                exit_price = price
                reason = "MODEL_EXIT"

            if exit_price is not None:
                gross = position.qty * exit_price
                exit_fee = gross * cfg.fee_rate
                cash += gross - exit_fee
                pnl = (exit_price - position.entry) * position.qty
                total_fees = (position.entry * position.qty * cfg.fee_rate) + exit_fee
                trades.append({
                    "entry_time": position.entry_time,
                    "exit_time": ts,
                    "entry": position.entry,
                    "exit": exit_price,
                    "qty": position.qty,
                    "pnl_before_fees": pnl,
                    "fees": total_fees,
                    "pnl_net": pnl - total_fees,
                    "reason": reason,
                })
                position = None

        if position is None and (not halted_today) and prob >= cfg.min_buy_probability:
            decision = size_position(
                equity=cash,
                price=price,
                atr_value=atr,
                risk_per_trade=cfg.risk_per_trade,
                stop_loss_atr=cfg.stop_loss_atr,
                max_position_pct=cfg.max_position_pct,
            )
            if decision.allowed:
                qty = decision.quantity
                cost = qty * price
                entry_fee = cost * cfg.fee_rate
                if cost + entry_fee <= cash:
                    cash -= (cost + entry_fee)
                    position = Position(
                        qty=qty,
                        entry=price,
                        stop=price - atr * cfg.stop_loss_atr,
                        target=price + atr * cfg.take_profit_atr,
                        entry_time=ts,
                    )

        equity = cash + (position.qty * price if position else 0)
        equity_curve.append({"timestamp": ts, "equity": equity})

    curve = pd.DataFrame(equity_curve)
    trade_df = pd.DataFrame(trades)

    final_equity = float(curve["equity"].iloc[-1]) if len(curve) else cash
    total_return = final_equity / (starting_cash or cfg.starting_cash) - 1

    peak = curve["equity"].cummax() if len(curve) else pd.Series(dtype=float)
    drawdown = (curve["equity"] / peak - 1) if len(curve) else pd.Series(dtype=float)
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    if len(trade_df):
        wins = int((trade_df["pnl_net"] > 0).sum())
        win_rate = wins / len(trade_df)
        gross_profit = trade_df.loc[trade_df["pnl_net"] > 0, "pnl_net"].sum()
        gross_loss = -trade_df.loc[trade_df["pnl_net"] < 0, "pnl_net"].sum()
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    else:
        win_rate = 0.0
        profit_factor = 0.0

    summary = {
        "starting_equity": starting_cash or cfg.starting_cash,
        "final_equity": round(final_equity, 2),
        "return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "trades": len(trade_df),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
    }
    return summary, trade_df, curve
