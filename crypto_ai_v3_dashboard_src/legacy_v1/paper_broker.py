from dataclasses import dataclass

@dataclass
class PaperPosition:
    symbol: str
    qty: float
    entry_price: float
    stop: float
    target: float

class PaperBroker:
    def __init__(self, starting_cash: float = 10_000):
        self.cash = starting_cash
        self.positions: dict[str, PaperPosition] = {}
        self.realized_pnl = 0.0

    def buy(self, symbol: str, qty: float, price: float, stop: float, target: float, fee_rate: float):
        if symbol in self.positions:
            return False, "Position already open"
        total = qty * price
        fee = total * fee_rate
        if total + fee > self.cash:
            return False, "Insufficient paper cash"
        self.cash -= total + fee
        self.positions[symbol] = PaperPosition(symbol, qty, price, stop, target)
        return True, "PAPER BUY"

    def sell(self, symbol: str, price: float, fee_rate: float):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return False, "No open position"
        proceeds = pos.qty * price
        fee = proceeds * fee_rate
        self.cash += proceeds - fee
        pnl = (price - pos.entry_price) * pos.qty - fee
        self.realized_pnl += pnl
        return True, f"PAPER SELL | approx PnL={pnl:.2f}"
