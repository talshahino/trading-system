//+------------------------------------------------------------------+
//| TradingSystem.mq5 - automatic demo robot for the FTMO account     |
//| Strategies (each has its own magic number, can be switched off):  |
//|   TV01 Gold Pro      4H (UTC bars), stop 2.25 ATR, trailing 2.25 ATR |
//|   TV04 Golden Edge   4H (UTC bars), stop 1.5 ATR, target 2 ATR,   |
//|                      break-even (+0.48%) at 69% of the target      |
//| Same rules as the backtest: signal at the 4-hour bar close, entry |
//| at the next open, 1% of $100,000 risk per trade, one position per |
//| strategy per instrument, no entries 5 min around high-impact news.|
//| Shared risk rule for the whole account: no new trade if today's   |
//| loss + every open stop + the new stop could pass 90% of the daily |
//| limit, or the total limit. Emergency: floating daily loss $4,500  |
//| -> close all until tomorrow (Prague midnight); equity <= $91,000  |
//| -> close all and stop.                                            |
//+------------------------------------------------------------------+
#property copyright "trading-system"
#property version   "1.10"

#include <Trade/Trade.mqh>

input string Strategies   = "TV01,TV04";
input string Symbols      = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,XAUUSD,XAGUSD,USOIL.cash";
input double InitialBalance = 100000;   // FTMO account size
input double RiskPct      = 1.0;        // % of the initial balance per trade
input double DailyPct     = 5.0;        // FTMO maximum daily loss
input double MaxPct       = 10.0;       // FTMO maximum loss
input int    NewsMinutes  = 5;          // no entries this many minutes before/after high-impact news
input int    H1History    = 4000;       // 1-hour bars used to build the 4-hour bars (indicator warm-up)
input string GitHubToken  = "";         // fine-grained token (this repository, Contents: read & write) - for the dashboard
input string GitHubRepo   = "talshahino/trading-system";
input int    PushMinutes  = 30;         // how often the account state is sent to the dashboard

CTrade trade;
string syms[];
bool   useTV01 = false, useTV04 = false;
const long MAGIC_TV01 = 260001;
const long MAGIC_TV04 = 260004;
string gvPrefix;

//---------------- time helpers ----------------
datetime NthSunday(int year, int month, int n)          // n = 1.. ; n = -1 -> last Sunday
{
   MqlDateTime t; ZeroMemory(t); t.year = year; t.mon = month; t.day = 1;
   datetime d = StructToTime(t);
   if(n > 0)
   {
      TimeToStruct(d, t);
      int add = (7 - t.day_of_week) % 7;
      return d + (add + 7 * (n - 1)) * 86400;
   }
   t.mon = month + 1; if(t.mon > 12) { t.mon = 1; t.year++; }
   datetime last = StructToTime(t) - 86400;
   TimeToStruct(last, t);
   return last - t.day_of_week * 86400;
}
bool UsDst(datetime utc)
{
   MqlDateTime t; TimeToStruct(utc, t);
   datetime a = NthSunday(t.year, 3, 2) + 7 * 3600, b = NthSunday(t.year, 11, 1) + 6 * 3600;
   return utc >= a && utc < b;
}
bool EuDst(datetime utc)
{
   MqlDateTime t; TimeToStruct(utc, t);
   datetime a = NthSunday(t.year, 3, -1) + 3600, b = NthSunday(t.year, 10, -1) + 3600;
   return utc >= a && utc < b;
}
int ServerBaseOffset()                                    // server offset in hours, without US DST
{
   int now = (int)MathRound((double)(TimeTradeServer() - TimeGMT()) / 3600.0);
   return now - (UsDst(TimeGMT()) ? 1 : 0);
}
datetime ServerToUtc(datetime s, int base)
{
   datetime guess = s - base * 3600;
   return s - (base + (UsDst(guess) ? 1 : 0)) * 3600;
}
int PragueDay()
{
   datetime p = TimeGMT() + (EuDst(TimeGMT()) ? 7200 : 3600);
   return (int)(p / 86400);
}

//---------------- 4-hour UTC bars built from 1-hour bars ----------------
struct Bars4
{
   datetime t[]; double o[], h[], l[], c[], v[];
   int n;
   datetime nextOpen;      // UTC open of the first 1-hour bar after the last completed 4-hour bar (0 = market not open yet)
};
bool Build4H(string s, Bars4 &b)
{
   MqlRates r[];
   ArraySetAsSeries(r, false);
   int got = CopyRates(s, PERIOD_H1, 0, H1History, r);
   if(got < 1000) return false;
   int base = ServerBaseOffset();
   datetime nowUtc = TimeGMT();
   ArrayResize(b.t, got); ArrayResize(b.o, got); ArrayResize(b.h, got);
   ArrayResize(b.l, got); ArrayResize(b.c, got); ArrayResize(b.v, got);
   b.n = 0; b.nextOpen = 0;
   for(int i = 0; i < got; i++)
   {
      datetime u = ServerToUtc(r[i].time, base);
      datetime bucket = u - (u % (4 * 3600));
      if(bucket + 4 * 3600 > nowUtc) { b.nextOpen = u; break; }   // only completed 4-hour bars
      if(b.n > 0 && b.t[b.n - 1] == bucket)
      {
         int k = b.n - 1;
         b.h[k] = MathMax(b.h[k], r[i].high); b.l[k] = MathMin(b.l[k], r[i].low);
         b.c[k] = r[i].close; b.v[k] += (double)r[i].tick_volume;
      }
      else
      {
         int k = b.n++;
         b.t[k] = bucket; b.o[k] = r[i].open; b.h[k] = r[i].high; b.l[k] = r[i].low;
         b.c[k] = r[i].close; b.v[k] = (double)r[i].tick_volume;
      }
   }
   return b.n > 300;
}

//---------------- indicators (same formulas as the backtest) ----------------
void Ema(const double &x[], int n, int len, double &out[])
{
   ArrayResize(out, n); double a = 2.0 / (len + 1);
   out[0] = x[0];
   for(int i = 1; i < n; i++) out[i] = a * x[i] + (1 - a) * out[i - 1];
}
void Rma(const double &x[], int n, int len, double &out[])
{
   ArrayResize(out, n); double a = 1.0 / len;
   out[0] = x[0];
   for(int i = 1; i < n; i++) out[i] = a * x[i] + (1 - a) * out[i - 1];
}
void Atr(Bars4 &b, int len, double &out[])
{
   double tr[]; ArrayResize(tr, b.n);
   for(int i = 0; i < b.n; i++)
   {
      double pc = (i == 0 ? b.c[0] : b.c[i - 1]);
      tr[i] = MathMax(b.h[i] - b.l[i], MathMax(MathAbs(b.h[i] - pc), MathAbs(b.l[i] - pc)));
   }
   Rma(tr, b.n, len, out);
}
void Rsi(Bars4 &b, int len, double &out[])
{
   double up[], dn[], ru[], rd[]; ArrayResize(up, b.n); ArrayResize(dn, b.n);
   for(int i = 0; i < b.n; i++)
   {
      double d = (i == 0 ? 0 : b.c[i] - b.c[i - 1]);
      up[i] = MathMax(d, 0); dn[i] = MathMax(-d, 0);
   }
   Rma(up, b.n, len, ru); Rma(dn, b.n, len, rd);
   ArrayResize(out, b.n);
   for(int i = 0; i < b.n; i++) out[i] = (rd[i] == 0 ? 100 : 100 - 100 / (1 + ru[i] / rd[i]));
}
void Macd(Bars4 &b, int f, int s, int g, double &m[], double &sig[])
{
   double ef[], es[]; Ema(b.c, b.n, f, ef); Ema(b.c, b.n, s, es);
   ArrayResize(m, b.n);
   for(int i = 0; i < b.n; i++) m[i] = ef[i] - es[i];
   Ema(m, b.n, g, sig);
}

//---------------- news filter (MT5 economic calendar) ----------------
void CurrenciesOf(string s, string &cur[])
{
   ArrayResize(cur, 0);
   string u = s; StringToUpper(u);
   if(StringFind(u, "XAU") == 0 || StringFind(u, "XAG") == 0) { ArrayResize(cur, 1); cur[0] = "USD"; return; }
   if(StringFind(u, "OIL") >= 0 || StringFind(u, "WTI") >= 0) { ArrayResize(cur, 2); cur[0] = "USD"; cur[1] = "CAD"; return; }
   ArrayResize(cur, 2); cur[0] = StringSubstr(u, 0, 3); cur[1] = StringSubstr(u, 3, 3);
}
bool NewsNear(string s)
{
   string cur[]; CurrenciesOf(s, cur);
   datetime now = TimeTradeServer();
   for(int k = 0; k < ArraySize(cur); k++)
   {
      MqlCalendarValue vals[];
      if(CalendarValueHistory(vals, now - NewsMinutes * 60, now + NewsMinutes * 60, NULL, cur[k]) <= 0) continue;
      for(int i = 0; i < ArraySize(vals); i++)
      {
         MqlCalendarEvent ev;
         if(CalendarEventById(vals[i].event_id, ev) && ev.importance == CALENDAR_IMPORTANCE_HIGH)
            return true;
      }
   }
   return false;
}

//---------------- account guard ----------------
double DayStart()
{
   string kd = gvPrefix + "day", kv = gvPrefix + "daystart";
   int today = PragueDay();
   if(!GlobalVariableCheck(kd) || (int)GlobalVariableGet(kd) != today)
   {
      GlobalVariableSet(kd, today);
      GlobalVariableSet(kv, MathMax(AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY)));
      GlobalVariableDel(gvPrefix + "halt_day");
   }
   return GlobalVariableGet(kv);
}
double WorstEquity()                                      // balance if every open position hits its stop
{
   double w = AccountInfoDouble(ACCOUNT_BALANCE);
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(!PositionSelectByTicket(tk)) continue;
      string s = PositionGetString(POSITION_SYMBOL);
      double sl = PositionGetDouble(POSITION_SL), vol = PositionGetDouble(POSITION_VOLUME);
      double op = PositionGetDouble(POSITION_PRICE_OPEN);
      long type = PositionGetInteger(POSITION_TYPE);
      if(sl <= 0) return -1e18;                             // a position without a stop: unknown risk
      double p = 0;
      if(!OrderCalcProfit(type == POSITION_TYPE_BUY ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, s, vol, op, sl, p)) return -1e18;
      w += p + PositionGetDouble(POSITION_SWAP);
   }
   return w;
}
void CloseAll(string why)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(PositionSelectByTicket(tk)) trade.PositionClose(tk);
   }
   Journal("ALL", "-", 0, 0, 0, 0, 0, why);
}
bool Halted()
{
   if(GlobalVariableCheck(gvPrefix + "halt_forever")) return true;
   return GlobalVariableCheck(gvPrefix + "halt_day") && (int)GlobalVariableGet(gvPrefix + "halt_day") == PragueDay();
}
void Guard()
{
   double ds = DayStart(), eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq <= InitialBalance * (1 - MaxPct / 100) + InitialBalance * RiskPct / 100)
   {
      if(!GlobalVariableCheck(gvPrefix + "halt_forever")) { CloseAll("STOP: equity near the total limit"); GlobalVariableSet(gvPrefix + "halt_forever", 1); }
      return;
   }
   if(eq <= ds - InitialBalance * DailyPct / 100 * 0.9 && !Halted())
   {
      CloseAll("STOP FOR TODAY: floating daily loss near the limit");
      GlobalVariableSet(gvPrefix + "halt_day", PragueDay());
   }
}
bool RiskAllows(double newRisk)
{
   double ds = DayStart(), worst = WorstEquity();
   if(worst - newRisk < ds - InitialBalance * DailyPct / 100 * 0.9) return false;
   if(worst - newRisk <= InitialBalance * (1 - MaxPct / 100)) return false;
   return true;
}

//---------------- journal (for the weekly comparison with the backtest) ----------------
void Journal(string strat, string s, int dir, double px, double sl, double tp, double lots, string note)
{
   int f = FileOpen("TradingSystem_journal.csv", FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');
   if(f == INVALID_HANDLE) return;
   if(FileSize(f) == 0) FileWrite(f, "utc_time", "strategy", "symbol", "dir", "price", "stop", "target", "lots", "note");
   FileSeek(f, 0, SEEK_END);
   FileWrite(f, TimeToString(TimeGMT(), TIME_DATE | TIME_MINUTES), strat, s, dir, px, sl, tp, lots, note);
   FileClose(f);
}

//---------------- positions ----------------
bool HasPosition(string s, long magic, ulong &ticket)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(PositionSelectByTicket(tk) && PositionGetString(POSITION_SYMBOL) == s && PositionGetInteger(POSITION_MAGIC) == magic)
      { ticket = tk; return true; }
   }
   return false;
}
double LotsFor(string s, int dir, double entry, double sl, double riskUsd)
{
   double p = 0;
   if(!OrderCalcProfit(dir > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, s, 1.0, entry, sl, p) || p >= 0) return 0;
   double lots = riskUsd / (-p);
   double step = SymbolInfoDouble(s, SYMBOL_VOLUME_STEP), mn = SymbolInfoDouble(s, SYMBOL_VOLUME_MIN), mx = SymbolInfoDouble(s, SYMBOL_VOLUME_MAX);
   lots = MathFloor(lots / step) * step;
   if(lots < mn) return 0;
   return MathMin(lots, mx);
}
void Open(string strat, long magic, string s, int dir, double stopDist, double tpDist)
{
   ulong tk;
   if(HasPosition(s, magic, tk)) return;
   if(Halted()) { Journal(strat, s, dir, 0, 0, 0, 0, "skipped: trading stopped"); return; }
   if(NewsNear(s)) { Journal(strat, s, dir, 0, 0, 0, 0, "skipped: high-impact news"); return; }
   double risk = InitialBalance * RiskPct / 100;
   if(!RiskAllows(risk)) { Journal(strat, s, dir, 0, 0, 0, 0, "skipped: shared risk rule"); return; }
   int dg = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   double px = dir > 0 ? SymbolInfoDouble(s, SYMBOL_ASK) : SymbolInfoDouble(s, SYMBOL_BID);
   double sl = NormalizeDouble(px - dir * stopDist, dg);
   double tp = tpDist > 0 ? NormalizeDouble(px + dir * tpDist, dg) : 0;
   double lots = LotsFor(s, dir, px, sl, risk);
   if(lots <= 0) { Journal(strat, s, dir, px, sl, tp, 0, "skipped: lot size below minimum"); return; }
   trade.SetExpertMagicNumber(magic);
   bool ok = dir > 0 ? trade.Buy(lots, s, 0, sl, tp, strat) : trade.Sell(lots, s, 0, sl, tp, strat);
   Journal(strat, s, dir, px, sl, tp, lots, ok ? "opened" : "open failed " + IntegerToString(trade.ResultRetcode()));
}
void MoveStop(ulong tk, double newSl, string strat, string s)
{
   if(!PositionSelectByTicket(tk)) return;
   long type = PositionGetInteger(POSITION_TYPE);
   int dir = type == POSITION_TYPE_BUY ? 1 : -1;
   double cur = PositionGetDouble(POSITION_SL), tp = PositionGetDouble(POSITION_TP);
   int dg = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   newSl = NormalizeDouble(newSl, dg);
   if(dir * (newSl - cur) <= 0) return;                     // stops only move in the trade's favour
   double px = dir > 0 ? SymbolInfoDouble(s, SYMBOL_BID) : SymbolInfoDouble(s, SYMBOL_ASK);
   if(dir * (px - newSl) <= 0) { trade.PositionClose(tk); Journal(strat, s, dir, px, newSl, tp, 0, "closed: new stop beyond price"); return; }
   trade.PositionModify(tk, newSl, tp);
}

//---------------- strategies (run once per completed 4-hour bar) ----------------
void RunTV01(string s, Bars4 &b)
{
   int n = b.n, i = n - 1;
   double e50[], e200[], m[], sg[], r[], a[];
   Ema(b.c, n, 50, e50); Ema(b.c, n, 200, e200); Macd(b, 12, 26, 9, m, sg); Rsi(b, 14, r); Atr(b, 14, a);
   ulong tk;
   if(HasPosition(s, MAGIC_TV01, tk))                        // trailing stop: best price since entry - 2.25 ATR
   {
      PositionSelectByTicket(tk);
      int dir = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
      datetime ent = ServerToUtc((datetime)PositionGetInteger(POSITION_TIME), ServerBaseOffset());
      double best = PositionGetDouble(POSITION_PRICE_OPEN), stop = PositionGetDouble(POSITION_SL);
      for(int k = 0; k < n; k++)
      {
         if(b.t[k] + 4 * 3600 <= ent) continue;
         best = dir > 0 ? MathMax(best, b.h[k]) : MathMin(best, b.l[k]);
         double ts = best - dir * 2.25 * a[k];
         stop = dir > 0 ? MathMax(stop, ts) : MathMin(stop, ts);
      }
      MoveStop(tk, stop, "TV01", s);
      return;
   }
   bool up = m[i] > sg[i] && m[i - 1] <= sg[i - 1], dn = m[i] < sg[i] && m[i - 1] >= sg[i - 1];
   if(e50[i] > e200[i] && b.c[i] > e200[i] && up && r[i] > 50 && r[i] < 70) Open("TV01", MAGIC_TV01, s, 1, 2.25 * a[i], 0);
   else if(e50[i] < e200[i] && b.c[i] < e200[i] && dn && r[i] > 30 && r[i] < 50) Open("TV01", MAGIC_TV01, s, -1, 2.25 * a[i], 0);
}
void RunTV04(string s, Bars4 &b)
{
   int n = b.n, i = n - 1;
   double m[], sg[], a[], ev6[], ev8[];
   Macd(b, 16, 26, 9, m, sg); Atr(b, 14, a); Ema(b.v, n, 6, ev6); Ema(b.v, n, 8, ev8);
   ulong tk;
   if(HasPosition(s, MAGIC_TV04, tk))                        // break-even: at 69% of the target, stop -> entry +0.48%
   {
      PositionSelectByTicket(tk);
      double op = PositionGetDouble(POSITION_PRICE_OPEN), tp = PositionGetDouble(POSITION_TP);
      datetime ent = ServerToUtc((datetime)PositionGetInteger(POSITION_TIME), ServerBaseOffset());
      double best = op;
      for(int k = 0; k < n; k++) if(b.t[k] + 4 * 3600 > ent) best = MathMax(best, b.h[k]);
      if(tp > op && best - op >= 0.69 * (tp - op)) MoveStop(tk, op * 1.0048, "TV04", s);
      return;
   }
   double h1 = m[i] - sg[i], h0 = m[i - 1] - sg[i - 1];
   if(h1 > 0 && h0 <= 0 && ev6[i] - ev8[i] > 0 && b.v[i] > 0.5 * b.v[i - 1]) Open("TV04", MAGIC_TV04, s, 1, 1.5 * a[i], 2.0 * a[i]);
}

//---------------- symbol names differ between servers (XAGUSD / XAGUSD.x / USOIL.cash / WTI ...) ----------------
string ResolveSymbol(string want)
{
   if(SymbolInfoInteger(want, SYMBOL_DIGITS) > 0 || SymbolSelect(want, true)) return want;
   string w = want; StringToUpper(w);
   string keys[];
   if(StringFind(w, "OIL") >= 0 || StringFind(w, "WTI") >= 0) { ArrayResize(keys, 3); keys[0] = "USOIL"; keys[1] = "WTI"; keys[2] = "CL"; }
   else { ArrayResize(keys, 1); keys[0] = StringSubstr(w, 0, 6); }
   for(int k = 0; k < ArraySize(keys); k++)
      for(int i = 0; i < SymbolsTotal(false); i++)
      {
         string nm = SymbolName(i, false), u = nm; StringToUpper(u);
         if(StringFind(u, keys[k]) == 0 && StringFind(u, "BRENT") < 0 && StringFind(u, "UKOIL") < 0) return nm;
      }
   return "";
}


//---------------- dashboard feed: account state -> GitHub (branch "live", file live/state.json) ----------------
datetime lastPush = 0;
string Esc(string x) { StringReplace(x, "\\", "\\\\"); StringReplace(x, "\"", "\\\""); StringReplace(x, "\n", " "); StringReplace(x, "\r", " "); return x; }
string Num(double v, int d = 2) { return DoubleToString(v, d); }
string Iso(datetime utc) { string t = TimeToString(utc, TIME_DATE | TIME_SECONDS); StringReplace(t, ".", "-"); StringReplace(t, " ", "T"); return t + "Z"; }
string StratOf(long magic) { return magic == MAGIC_TV01 ? "TV01" : (magic == MAGIC_TV04 ? "TV04" : "manual"); }

void RecordEquity()                                       // one point per push, kept on disk
{
   int f = FileOpen("TradingSystem_equity.csv", FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');
   if(f == INVALID_HANDLE) return;
   FileSeek(f, 0, SEEK_END);
   FileWrite(f, Iso(TimeGMT()), Num(AccountInfoDouble(ACCOUNT_BALANCE)), Num(AccountInfoDouble(ACCOUNT_EQUITY)));
   FileClose(f);
}
string EquityJson()
{
   string lines[]; int n = 0;
   int f = FileOpen("TradingSystem_equity.csv", FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
   if(f != INVALID_HANDLE)
   {
      while(!FileIsEnding(f)) { string ln = FileReadString(f); if(ln == "") continue; ArrayResize(lines, n + 1, 1000); lines[n++] = ln; }
      FileClose(f);
   }
   string out = ""; int from = MathMax(0, n - 4000);       // last ~80 days at 30-minute points
   for(int i = from; i < n; i++)
   {
      string p[]; if(StringSplit(lines[i], ',', p) < 3) continue;
      out += (out == "" ? "" : ",") + "{\"t\":\"" + p[0] + "\",\"balance\":" + p[1] + ",\"equity\":" + p[2] + "}";
   }
   return "[" + out + "]";
}
string TradesJson()                                       // closed positions, one row each
{
   HistorySelect(0, TimeCurrent() + 86400);
   int base = ServerBaseOffset();
   string out = "";
   for(int i = 0; i < HistoryDealsTotal(); i++)
   {
      ulong d = HistoryDealGetTicket(i);
      long entry = HistoryDealGetInteger(d, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY) continue;
      long pos = HistoryDealGetInteger(d, DEAL_POSITION_ID);
      double inPx = 0; datetime inT = 0; long magic = HistoryDealGetInteger(d, DEAL_MAGIC); double comm = 0;
      for(int j = 0; j < HistoryDealsTotal(); j++)
      {
         ulong e = HistoryDealGetTicket(j);
         if(HistoryDealGetInteger(e, DEAL_POSITION_ID) == pos && HistoryDealGetInteger(e, DEAL_ENTRY) == DEAL_ENTRY_IN)
         { inPx = HistoryDealGetDouble(e, DEAL_PRICE); inT = (datetime)HistoryDealGetInteger(e, DEAL_TIME); magic = HistoryDealGetInteger(e, DEAL_MAGIC); comm += HistoryDealGetDouble(e, DEAL_COMMISSION); }
      }
      long type = HistoryDealGetInteger(d, DEAL_TYPE);       // the closing deal is opposite to the position
      double net = HistoryDealGetDouble(d, DEAL_PROFIT) + HistoryDealGetDouble(d, DEAL_SWAP) + HistoryDealGetDouble(d, DEAL_COMMISSION) + comm;
      out += (out == "" ? "" : ",") + "{\"id\":" + IntegerToString(pos) + ",\"strategy\":\"" + StratOf(magic) + "\",\"symbol\":\"" + Esc(HistoryDealGetString(d, DEAL_SYMBOL)) +
             "\",\"dir\":\"" + (type == DEAL_TYPE_SELL ? "buy" : "sell") + "\",\"volume\":" + Num(HistoryDealGetDouble(d, DEAL_VOLUME)) +
             ",\"open_utc\":\"" + (inT > 0 ? Iso(ServerToUtc(inT, base)) : "") + "\",\"open_price\":" + Num(inPx, 5) +
             ",\"close_utc\":\"" + Iso(ServerToUtc((datetime)HistoryDealGetInteger(d, DEAL_TIME), base)) + "\",\"close_price\":" + Num(HistoryDealGetDouble(d, DEAL_PRICE), 5) +
             ",\"profit\":" + Num(net) + "}";
   }
   return "[" + out + "]";
}
string PositionsJson()
{
   int base = ServerBaseOffset();
   string out = "";
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(!PositionSelectByTicket(tk)) continue;
      out += (out == "" ? "" : ",") + "{\"id\":" + IntegerToString((long)tk) + ",\"strategy\":\"" + StratOf(PositionGetInteger(POSITION_MAGIC)) +
             "\",\"symbol\":\"" + Esc(PositionGetString(POSITION_SYMBOL)) + "\",\"dir\":\"" + (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "buy" : "sell") +
             "\",\"volume\":" + Num(PositionGetDouble(POSITION_VOLUME)) + ",\"open_utc\":\"" + Iso(ServerToUtc((datetime)PositionGetInteger(POSITION_TIME), base)) +
             "\",\"open_price\":" + Num(PositionGetDouble(POSITION_PRICE_OPEN), 5) + ",\"stop\":" + Num(PositionGetDouble(POSITION_SL), 5) +
             ",\"target\":" + Num(PositionGetDouble(POSITION_TP), 5) + ",\"profit\":" + Num(PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP)) + "}";
   }
   return "[" + out + "]";
}
string JournalJson()                                      // the robot's last 300 actions (opened / skipped and why)
{
   string lines[]; int n = 0;
   int f = FileOpen("TradingSystem_journal.csv", FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
   if(f != INVALID_HANDLE)
   {
      while(!FileIsEnding(f)) { string ln = FileReadString(f); if(ln == "" || StringFind(ln, "utc_time") == 0) continue; ArrayResize(lines, n + 1, 500); lines[n++] = ln; }
      FileClose(f);
   }
   string out = "";
   for(int i = MathMax(0, n - 300); i < n; i++) out += (out == "" ? "" : ",") + "\"" + Esc(lines[i]) + "\"";
   return "[" + out + "]";
}
bool Http(string method, string url, string body, string &resp, int &code)
{
   string hdr = "Authorization: Bearer " + GitHubToken + "\r\nAccept: application/vnd.github+json\r\nUser-Agent: TradingSystem-EA\r\nContent-Type: application/json\r\n";
   char data[], res[]; string rh;
   if(body != "") { int k = StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8); ArrayResize(data, k - 1); }
   ResetLastError();
   code = WebRequest(method, url, hdr, 15000, data, res, rh);
   resp = CharArrayToString(res, 0, WHOLE_ARRAY, CP_UTF8);
   if(code == -1) { Print("Dashboard: web request failed (", GetLastError(), "). Add https://api.github.com under Tools > Options > Expert Advisors > Allow WebRequest."); return false; }
   return true;
}
void PushState()
{
   if(GitHubToken == "" || TimeGMT() - lastPush < PushMinutes * 60) return;
   lastPush = TimeGMT();
   RecordEquity();
   string json = "{\"updated_utc\":\"" + Iso(TimeGMT()) + "\",\"server\":\"" + Esc(AccountInfoString(ACCOUNT_SERVER)) +
                 "\",\"strategies\":\"" + Esc(Strategies) + "\",\"initial\":" + Num(InitialBalance) +
                 ",\"balance\":" + Num(AccountInfoDouble(ACCOUNT_BALANCE)) + ",\"equity\":" + Num(AccountInfoDouble(ACCOUNT_EQUITY)) +
                 ",\"day_start\":" + Num(DayStart()) + ",\"halted_today\":" + (Halted() ? "true" : "false") +
                 ",\"halted_forever\":" + (GlobalVariableCheck(gvPrefix + "halt_forever") ? "true" : "false") +
                 ",\"positions\":" + PositionsJson() + ",\"trades\":" + TradesJson() + ",\"equity_curve\":" + EquityJson() +
                 ",\"journal\":" + JournalJson() + "}";
   uchar raw[], key[], b64[];
   int k = StringToCharArray(json, raw, 0, WHOLE_ARRAY, CP_UTF8); ArrayResize(raw, k - 1);
   if(CryptEncode(CRYPT_BASE64, raw, key, b64) <= 0) return;
   string content = CharArrayToString(b64, 0, WHOLE_ARRAY, CP_ACP);
   StringReplace(content, "\r", ""); StringReplace(content, "\n", "");
   string url = "https://api.github.com/repos/" + GitHubRepo + "/contents/live/state.json";
   string resp; int code;
   string sha = "";
   if(Http("GET", url + "?ref=live", "", resp, code) && code == 200)
   {
      int a = StringFind(resp, "\"sha\":\"");
      if(a >= 0) { int b = StringFind(resp, "\"", a + 7); sha = StringSubstr(resp, a + 7, b - a - 7); }
   }
   string body = "{\"message\":\"account state\",\"branch\":\"live\",\"content\":\"" + content + "\"" + (sha != "" ? ",\"sha\":\"" + sha + "\"" : "") + "}";
   if(Http("PUT", url, body, resp, code) && (code == 200 || code == 201)) Print("Dashboard: state sent");
   else if(code != -1) Print("Dashboard: GitHub answered ", code, " ", StringSubstr(resp, 0, 200));
}

//---------------- main loop ----------------
int OnInit()
{
   gvPrefix = "TS_" + IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)) + "_";
   string st[]; StringSplit(Strategies, ',', st);
   for(int k = 0; k < ArraySize(st); k++) { StringTrimLeft(st[k]); StringTrimRight(st[k]); if(st[k] == "TV01") useTV01 = true; if(st[k] == "TV04") useTV04 = true; }
   StringSplit(Symbols, ',', syms);
   for(int k = 0; k < ArraySize(syms); k++)
   {
      StringTrimLeft(syms[k]); StringTrimRight(syms[k]);
      syms[k] = ResolveSymbol(syms[k]);
      if(syms[k] == "" || !SymbolSelect(syms[k], true)) Print("Symbol not found on this server: ", syms[k]);
      else Print("Trading symbol: ", syms[k]);
   }
   trade.SetDeviationInPoints(30);
   EventSetTimer(20);
   DayStart();
   Print("TradingSystem started. Strategies: ", Strategies, "  symbols: ", Symbols);
   return INIT_SUCCEEDED;
}
void OnDeinit(const int reason) { EventKillTimer(); }
void OnTick() {}
void OnTimer()
{
   Guard();
   PushState();
   for(int k = 0; k < ArraySize(syms); k++)
   {
      string s = syms[k];
      if(!SymbolInfoInteger(s, SYMBOL_SELECT)) continue;
      Bars4 b;
      if(!Build4H(s, b)) continue;
      string key = gvPrefix + "last_" + s;
      datetime last = b.t[b.n - 1];
      if(!GlobalVariableCheck(key)) { GlobalVariableSet(key, (double)last); continue; }   // first start: wait for the next bar
      if((datetime)GlobalVariableGet(key) >= last) continue;
      // like the backtest: act at the first open after the bar closed (after a weekend: Sunday's open);
      // if that open was more than 30 minutes ago (robot was off), skip this bar
      if(b.nextOpen == 0) continue;
      if(TimeGMT() - b.nextOpen > 30 * 60) { GlobalVariableSet(key, (double)last); continue; }
      if(SymbolInfoInteger(s, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_FULL) continue;
      if(useTV01) RunTV01(s, b);
      if(useTV04) RunTV04(s, b);
      GlobalVariableSet(key, (double)last);
   }
}
//+------------------------------------------------------------------+
