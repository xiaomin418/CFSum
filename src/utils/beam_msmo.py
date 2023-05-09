class Beam(object):
  def __init__(self, tokens, trigrams, log_probs, state, context, coverage):
    self.tokens = tokens
    self.trigrams = trigrams
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    ng=6
    if len(self.tokens)>ng-1:
        cur_trigram = tuple(w for w in self.tokens[-ng+1:])+(token,)
        if cur_trigram in self.trigrams:
            log_prob =  -1e20
        self.trigrams.append(cur_trigram)
    elif len(self.tokens)==ng-1:
        cur_trigram = tuple(w for w in self.tokens[-ng+1:])+(token,)
        self.trigrams.append(cur_trigram)
    else:
        pass
    return Beam(tokens = self.tokens + [token],
                trigrams = self.trigrams,
                log_probs = self.log_probs + [log_prob],
                state = state,
                context = context,
                coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)
