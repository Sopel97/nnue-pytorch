#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#if defined (__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif
#endif

using namespace binpack;
using namespace chess;

static Square orient(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return sq.flippedVertically().flippedHorizontally();
    }
}

static Square orient_flip(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        return sq.flippedVertically();
    }
}

struct HalfKP {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
        auto ksq = pos.kingSquare(color);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int features_unordered[32];
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            features_unordered[j++] = feature_index(color, orient(color, ksq), sq, p);
        }
        std::sort(features_unordered, features_unordered + j);
        for (int k = 0; k < j; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
        return INPUTS;
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_SQ * HalfKP::NUM_PT;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto counter_before = counter;
        int offset = HalfKP::fill_features_sparse(i, e, features, values, counter, color);
        auto& pos = e.pos;
        {
            auto num_added_features = counter - counter_before;
            // king square factor
            auto ksq = pos.kingSquare(color);
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = offset + static_cast<int>(orient(color, ksq));
            values[counter] = static_cast<float>(num_added_features);
            counter += 1;
        }
        offset += K_INPUTS;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        int features_unordered[32];
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            features_unordered[j++] = offset + (p_idx * HalfKP::NUM_SQ) + static_cast<int>(orient(color, sq));
        }
        std::sort(features_unordered, features_unordered + j);
        for (int k = 0; k < j; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
    }
};

struct HalfKA {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 12;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int features_unordered[32];
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            features_unordered[j++] = feature_index(color, orient_flip(color, ksq), sq, p);
        }
        std::sort(features_unordered, features_unordered + j);
        for (int k = 0; k < j; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
        return INPUTS;
    }
};

struct HalfKAFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
    static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto counter_before = counter;
        int offset = HalfKA::fill_features_sparse(i, e, features, values, counter, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        int features_unordered[32];
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            features_unordered[j++] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
        }
        std::sort(features_unordered, features_unordered + j);
        for (int k = 0; k < j; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
    }
};

struct HalfKAS2v1 {
    /*
        2 halfka buckets
        all KK features are packed into one NUM_SQ * NUM_SQ because
        they are disjoint
    */

    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 11;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ * 2;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p, Bitboard special)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        if (p_idx == 11)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        auto special_offset = special.isSet(sq) * (INPUTS / 2);
        auto oriented_sq = static_cast<int>(orient_flip(color, sq));
        return oriented_sq + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES + special_offset;
    }

    static Bitboard get_special_pawns(const Position& pos) {
        static const EnumArray<Square, Bitboard> forward_white = [](){
            EnumArray<Square, Bitboard> arr;
            for (Square s : values<Square>())
            {
                Bitboard bb{};
                for (int i = 1; i <= 6; ++i)
                {
                    bb |= Bitboard::square(s).shifted(Offset{0, i});
                }
                arr[s] = bb;
            }
            return arr;
        }();

        static const EnumArray<Square, Bitboard> forward_black = [](){
            EnumArray<Square, Bitboard> arr;
            for (Square s : values<Square>())
            {
                Bitboard bb{};
                for (int i = 1; i <= 6; ++i)
                {
                    bb |= Bitboard::square(s).shifted(Offset{0, -i});
                }
                arr[s] = bb;
            }
            return arr;
        }();

        static const EnumArray<Square, Bitboard> forward_span_white = [](){
            EnumArray<Square, Bitboard> arr;
            for (Square s : values<Square>())
            {
                Bitboard bb = forward_white[s];
                bb |= bb.shifted(Offset{1, 0});
                bb |= bb.shifted(Offset{-1, 0});
                arr[s] = bb;
            }
            return arr;
        }();

        static const EnumArray<Square, Bitboard> forward_span_black = [](){
            EnumArray<Square, Bitboard> arr;
            for (Square s : values<Square>())
            {
                Bitboard bb = forward_black[s];
                bb |= bb.shifted(Offset{1, 0});
                bb |= bb.shifted(Offset{-1, 0});
                arr[s] = bb;
            }
            return arr;
        }();

        const Bitboard white_pawns = pos.piecesBB(Piece(PieceType::Pawn, Color::White));
        const Bitboard black_pawns = pos.piecesBB(Piece(PieceType::Pawn, Color::Black));
        const Bitboard all_pawns = white_pawns | black_pawns;
        Bitboard special{};

        for (Square sq : white_pawns)
        {
            if (
                !((forward_white[sq] & all_pawns).any()
                  || (forward_span_white[sq] & black_pawns).any()))
            {
                special |= sq;
            }
        }

        for (Square sq : black_pawns)
        {
            if (
                !((forward_black[sq] & all_pawns).any()
                  || (forward_span_black[sq] & white_pawns).any()))
            {
                special |= sq;
            }
        }

        return special;
    }

    static Bitboard get_outpost_squares(const Position& pos)
    {
        const Bitboard white_pawns = pos.piecesBB(Piece(PieceType::Pawn, Color::White));
        const Bitboard black_pawns = pos.piecesBB(Piece(PieceType::Pawn, Color::Black));
        const Bitboard white_pawn_attacks = bb::pawnAttacks(white_pawns, Color::White);
        const Bitboard black_pawn_attacks = bb::pawnAttacks(black_pawns, Color::Black);

        const Bitboard outposts = ((white_pawn_attacks | white_pawns.shifted<0, -1>()) & ~black_pawn_attacks)
                                  | ((black_pawn_attacks | black_pawns.shifted<0, 1>()) & ~white_pawn_attacks);
        return outposts;
    }

    static Bitboard get_special_rooks(const Position& pos, Bitboard special_pawns)
    {
        const Bitboard white_pawns = pos.piecesBB(Piece(PieceType::Pawn, Color::White));
        const Bitboard black_pawns = pos.piecesBB(Piece(PieceType::Pawn, Color::Black));
        const Bitboard non_special_pawns = (white_pawns | black_pawns) & ~special_pawns;
        const Bitboard white_rooks = pos.piecesBB(Piece(PieceType::Rook, Color::White));
        const Bitboard black_rooks = pos.piecesBB(Piece(PieceType::Rook, Color::Black));
        const Bitboard occupied = pos.piecesBB();
        Bitboard special{};
        for (Square sq : white_rooks)
        {
            const auto filebb = Bitboard::file(sq.file());
            if (!((non_special_pawns & filebb).any())
                || (bb::fancy_magics::rookAttacks(sq, occupied) & white_rooks).any())
            {
                special |= sq;
            }
        }
        for (Square sq : black_rooks)
        {
            const auto filebb = Bitboard::file(sq.file());
            if (!((non_special_pawns & filebb).any())
                || (bb::fancy_magics::rookAttacks(sq, occupied) & black_rooks).any())
            {
                special |= sq;
            }
        }
        return special;
    }

    static Bitboard get_special_queens(const Position& pos)
    {
        const Bitboard white_queens = pos.piecesBB(Piece(PieceType::Queen, Color::White));
        const Bitboard black_queens = pos.piecesBB(Piece(PieceType::Queen, Color::Black));
        if (white_queens.count() != black_queens.count())
        {
            return white_queens | black_queens;
        }
        else
        {
            return Bitboard{};
        }
    }

    static Bitboard get_special_kings(const Position& pos)
    {
        static const EnumArray<Square, Bitboard> forward_white = [](){
            EnumArray<Square, Bitboard> arr;
            for (Square s : values<Square>())
            {
                Bitboard bb = Bitboard::square(s);
                for (int i = 1; i <= 3; ++i)
                {
                    bb |= bb.shifted(Offset{0, 1});
                }
                arr[s] = bb | bb.shifted<-1, 0>() | bb.shifted<1, 0>();
            }
            return arr;
        }();

        static const EnumArray<Square, Bitboard> forward_black = [](){
            EnumArray<Square, Bitboard> arr;
            for (Square s : values<Square>())
            {
                Bitboard bb = Bitboard::square(s);
                for (int i = 1; i <= 3; ++i)
                {
                    bb |= bb.shifted(Offset{0, -1});
                }
                arr[s] = bb | bb.shifted<-1, 0>() | bb.shifted<1, 0>();
            }
            return arr;
        }();

        const Bitboard white_pawns = pos.piecesBB(Piece(PieceType::Pawn, Color::White));
        const Bitboard black_pawns = pos.piecesBB(Piece(PieceType::Pawn, Color::Black));
        const Square king_white = pos.kingSquare(Color::White);
        const Square king_black = pos.kingSquare(Color::Black);

        Bitboard special{};
        if ((forward_white[king_white] & white_pawns).count() >= 3)
            special |= king_white;
        if ((forward_black[king_black] & black_pawns).count() >= 3)
            special |= king_black;
        return special;
    }

    static Bitboard get_special_squares(const Position& pos)
    {
        /*
            Squares are considered special if the piece on the square is special.
            Special may also mean worse than normal, this is just a bucketing scheme.

            Pawns:
                - passed
                    - no other pawn in front of the pawn
                    - no opponent pawns in front sides of the pawn

            Bishops & Bishops:
                - outpost
                    - defended by pawn and not attacked by pawn
                    - behind pawn of the same color and not attacked by pawn

            Rooks:
                - connected rooks
                - semiopen file
                - attacks/defends special pawn

            Queens:
                - queen imbalance
                    - one side has more queens than the other

            Kings:
                - safe king
                    - at least 3 pawns of the same color in the frontmost (span) of 9 squares and to the sides
        */

        Bitboard special{};
        const Bitboard special_pawns = get_special_pawns(pos);
        special |= special_pawns;
        special |= get_outpost_squares(pos) & (pos.piecesBB(Piece(PieceType::Bishop, Color::White))
                                               | pos.piecesBB(Piece(PieceType::Bishop, Color::Black))
                                               | pos.piecesBB(Piece(PieceType::Knight, Color::White))
                                               | pos.piecesBB(Piece(PieceType::Knight, Color::Black)));
        special |= get_special_rooks(pos, special_pawns);
        special |= get_special_queens(pos);
        special |= get_special_kings(pos);

        return special;
    }

    static int fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);
        auto special = get_special_squares(pos);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int features_unordered[32];
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            features_unordered[j++] = feature_index(color, orient_flip(color, ksq), sq, p, special);
        }
        std::sort(features_unordered, features_unordered + j);
        for (int k = 0; k < j; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
        return INPUTS;
    }
};

struct HalfKAS2v1Factorized {
    // Factorized features
    static constexpr int NUM_PT_VIRTUAL = 12;
    static constexpr int PIECE_INPUTS = HalfKAS2v1::NUM_SQ * NUM_PT_VIRTUAL;
    static constexpr int INPUTS = HalfKAS2v1::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAS2v1::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        auto counter_before = counter;
        int offset = HalfKAS2v1::fill_features_sparse(i, e, features, values, counter, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        int features_unordered[32];
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            features_unordered[j++] = offset + (p_idx * HalfKAS2v1::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
        }
        std::sort(features_unordered, features_unordered + j);
        for (int k = 0; k < j; ++k)
        {
            int idx = counter * 2;
            features[idx] = i;
            features[idx + 1] = features_unordered[k];
            values[counter] = 1.0f;
            counter += 1;
        }
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, float* values, int& counter, Color color)
    {
        T::fill_features_sparse(i, e, features, values, counter, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        psqt_indices = new int[size];
        layer_stack_indices = new int[size];

        num_active_white_features = 0;
        num_active_black_features = 0;

        std::memset(white, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));
        std::memset(black, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* psqt_indices;
    int* layer_stack_indices;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] psqt_indices;
        delete[] layer_stack_indices;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos.piecesBB().count() - 1) / 4;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        FeatureSet<Ts...>::fill_features_sparse(i, e, white, white_values, num_active_white_features, Color::White);
        FeatureSet<Ts...>::fill_features_sparse(i, e, black, black_values, num_active_black_features, Color::Black);
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename, cyclic, skipPredicate))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filename, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const char* filename, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new StorageT(FeatureSet{}, entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

extern "C" {

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic, bool filtered, int random_fen_skipping)
    {
        std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr;
        if (filtered || random_fen_skipping)
        {
            skipPredicate = [
                random_fen_skipping,
                prob = double(random_fen_skipping) / (random_fen_skipping + 1),
                filtered
                ](const TrainingDataEntry& e){

                auto do_skip = [&]() {
                    std::bernoulli_distribution distrib(prob);
                    auto& prng = rng::get_thread_local_rng();
                    return distrib(prng);
                };

                auto do_filter = [&]() {
                    return (e.isCapturingMove() || e.isInCheck());
                };

                static thread_local std::mt19937 gen(std::random_device{}());
                return (random_fen_skipping && do_skip()) || (filtered && do_filter());
            };
        }

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKP^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKAS2v1")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAS2v1>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKAS2v1^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAS2v1Factorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

}

/* benches */ //*
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP", 4, "10m_d3_q_2.binpack", 8192, true, false, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
//*/
